import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { postLabelsOpts, postDiscardOpts } from "../schemas/schemas";


module.exports = function(server: FastifyInstance, opts: unknown, done: () => void) {
  
  server.addHook("onRequest", async (request: FastifyRequest, reply: FastifyReply) => {
    try {
      request.log.info("routes check token");
      const token = request.appToken.get('data') || '';
      server.jwt.verify(token);
      request.log.info("routes appToken OK");
    } catch (err) {
      try{
        request.log.info("routes appToken BAD");
        const refreshToken = request.appRefresh.get('data') || '';
        request.log.info("routes appToken checking again");
        server.jwt.verify(refreshToken, (err, verified) => {
          if (err) {
            request.log.warn("routes Authorization not found in request.");
            return reply.view('login.ejs', { error: 'Sign in to access this resource.' });
          }
          return verified;
        });
        request.log.info("routes appRefresh BAD");
        request.appToken.set('data', refreshToken);
        request.appRefresh.set('data', refreshToken);
      } catch(err2) {
        request.log.warn("routes Authorization not found in request.");
        return reply.view('login.ejs', { error: 'Sign in to access this resource.' });
      }
    }
  });

  //------------------------------------------------------------------
  server.post('/discard', postDiscardOpts);

  //------------------------------------------------------------------
  server.post('/labels', postLabelsOpts);

  //------------------------------------------------------------------
  
  server.get('/', async (request: FastifyRequest, reply: FastifyReply) => {
    return reply.sendFile('index.html');
  });
  
  //------------------------------------------------------------------

  done();
}

//------------------------------------------------------------------
