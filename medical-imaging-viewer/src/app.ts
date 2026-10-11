import path from 'path';
import fastify, { FastifyInstance } from 'fastify';
import fastifyStatic from '@fastify/static';
import fastifyCors from '@fastify/cors';
import fastifySensible from '@fastify/sensible';
import fastifyHelmet from '@fastify/helmet';
import fastifyAutoload from '@fastify/autoload';
import fastifyJwt from '@fastify/jwt';
import fastifySecureSession, { Session } from '@fastify/secure-session';
import fastifyMysql, { MySQLPromisePool } from '@fastify/mysql';
import fastifyFormBody from '@fastify/formbody';
import fastifyView from '@fastify/view';
import { sendEcho } from './dimse/sendEcho';
import { startScp, shutdown } from './dimse/store';
import { clearCache } from './utils/fileHelper';
import { ConfParams, config } from './utils/config';
import configEnv from "./config";
import ejs from 'ejs';
import { Http2SecureServer, Http2ServerRequest, Http2ServerResponse } from 'http2';
import fs from 'fs';

const envToLogger = {
  development: {
    transport: {
      target: 'pino-pretty',
      options: {
        translateTime: 'yyyy-mm-dd HH:MM:ss',
        ignore: 'pid,hostname,reqId,responseTime,req,res',
        messageFormat: '{msg}',
      },
    },
  },
  production: true,
  test: false,
}

const server: FastifyInstance<Http2SecureServer, Http2ServerRequest, Http2ServerResponse> = fastify({
  http2: true,
  https: {
    allowHTTP1: true,
    key: fs.readFileSync(path.join(__dirname, '/certs/server.key')),
    cert: fs.readFileSync(path.join(__dirname, '/certs/server.crt')),
  },
  logger: envToLogger['development'] ?? true, // defaults to true if no entry matches in the map
  disableRequestLogging: true
});

server.register(fastifyJwt, {
  secret: configEnv.JWT_TOKEN,
  formatUser: function (user: any) {
    return {
      userId: user.userId,
      username: user.username,
      isAdmin: user.isAdmin
    }
  },
  // Global default decoding method options
  decode: { complete: false },
});

server.register(fastifySecureSession, [{
  // the name of the attribute decorated on the request-object, defaults to 'session'
  sessionName: configEnv.SESSION_NAME,
  // the name of the session cookie, defaults to value of sessionName
  cookieName: configEnv.COOKIE_NAME,
  // adapt this to point to the directory where secret-key is located
  key: fs.readFileSync(path.join(__dirname, 'certs/sessions/session-key1')),
  cookie: {
    maxAge: 60 * 5, // 5 minutes
    httpOnly: true,
    //signed: true,
    sameSite: true,
    path: '/',
    secure: true
  }
},
{
  // the name of the attribute decorated on the request-object, defaults to 'session'
  sessionName: configEnv.REFRESH_SESSION_NAME,
  // the name of the session cookie, defaults to value of sessionName
  cookieName: configEnv.REFRESH_COOKIE_NAME,
  // adapt this to point to the directory where secret-key is located
  key: fs.readFileSync(path.join(__dirname, 'certs/sessions/refresh-session-key1')),
  cookie: {
    maxAge: 60 * 60 * 2, // 2 hours
    httpOnly: true,
    //signed: true,
    sameSite: true,
    path: '/',
    secure: true
  }
}]);

server.register(fastifyMysql, {
  //type: 'connection',
  promise: true,
  host: configEnv.DB_HOST,
  port: configEnv.DB_PORT,
  user: configEnv.DB_USER,
  password: configEnv.DB_PWD,
  database: configEnv.DB_NAME
});

server.register(fastifyStatic, {
  root: path.join(__dirname, '../public'),
});

interface RefreshSessionData {
  data: string;
}
interface AccessSessionData {
  data: string;
}
declare module "fastify" {
  interface FastifyRequest {
    appRefresh: Session<RefreshSessionData>;
    appToken: Session<AccessSessionData>;
  }
  interface FastifyInstance {
    mysql: MySQLPromisePool;
  }
}

server.setNotFoundHandler(function (request, reply) {
  request.log.info("not found");
  //return reply.view('/includes/login.ejs', { error: "Recurso no encontrado." });
  return reply.redirect('/login');
});

// Register ejs engine
server.register(fastifyView, {
  engine: {
    ejs: ejs,
  },
  root: path.join(__dirname, 'views'),
});

server.register(fastifyFormBody);
server.register(fastifyCors, {});
server.register(fastifySensible);
server.register(fastifyHelmet, { contentSecurityPolicy: false });
//server.register(routesProtected)
//server.register(routesUnprotected)


server.register(fastifyAutoload, {
  dir: path.join(__dirname, 'routes'),
});

server.register(fastifyAutoload, {
  dir: path.join(__dirname, 'routesohif'),
});
server.register(fastifyAutoload, {
  dir: path.join(__dirname, 'routesohif'),
  options: { prefix: '/viewer' },
});

/*
server.register(routesProtected);
server.register(routesUnprotected);
server.register(ohifRoutesProtected);
*/
//------------------------------------------------------------------

// log exceptions
process.on('uncaughtException', (err) => {
  server.log.error('uncaught exception received:');
  server.log.error(err.stack);
});

//------------------------------------------------------------------

process.on('SIGINT', () => {
  server.log.info('shutting down web server...');
  server.close().then(()=>{
    server.log.info('webserver shutdown successfully');
    if (!config.get(ConfParams.C_GET)) {
      server.log.info('shutting down DICOM SCP server...');
      shutdown(server.log).then(()=>{
        server.log.info('DICOM SCP server shutdown successfully...');
      }, (err2) => {
        server.log.info('Error shutting down DICOM SCP server...' + err2);
      });
    }
    process.exit(1);
  }, (err) => {
    server.log.error('Error shutting down web server... ' + err);
  });
});

//------------------------------------------------------------------

const port = config.get(ConfParams.HTTP_PORT) as number;
const host = '0.0.0.0';

//server.register(fastifyHttpsRedirect, { httpPort: 80, httpsPort: port });

/**
 * Run the server!
 */
const start = async () => {
  try {
    server.log.info('starting...', port);
    // if not using c-get, start our scp
    if (!config.get(ConfParams.C_GET)) {
      startScp(server.log);
    }
    sendEcho(server.log);
    // running clear cache and setup for 2min checks
    clearCache(server.log);
    setInterval(async () => {
      clearCache(server.log).then(() => {
        server.log.info("clearCache was successfull");
      }).catch((reason) => {
        server.log.error("clearCache errored: " + reason);
      });
    }, 120000); // milliseconds (1500000 --> 25 mins)
    server.listen({ port: port, host: host })
  } catch (err) {
    server.log.error(err)
    //process.exit(1)
  }
}
start();

//------------------------------------------------------------------
