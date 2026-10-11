import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { loginOpts, postContactOpts } from "../schemas/schemas";
import config from '../config';

module.exports = function(server: FastifyInstance, opts: unknown, done: () => void) {

  server.get('/login', async (request: FastifyRequest, reply: FastifyReply) => {
    return reply.view('login.ejs')
  });
  
  //------------------------------------------------------------------

  server.get('/logout', async (request: FastifyRequest, reply: FastifyReply) => {
    request.appToken.delete()
    request.appRefresh.delete()
    //reply.clearCookie(config.REFRESH_TOKEN_NAME);
    return reply.redirect('/login')
  });
  
  //------------------------------------------------------------------
  
  server.get('/contact', async (request: FastifyRequest, reply: FastifyReply) => {
    return reply.view('contact.ejs', { returnPage: '/login' });
  });
  
  //------------------------------------------------------------------
  
  server.post('/login', loginOpts);
  
  //------------------------------------------------------------------

  server.post('/contact', postContactOpts);
  
  //------------------------------------------------------------------
  /*
  server.post('/hashpassword', async (request: FastifyRequest<{ Body: {password: string} }>, reply: FastifyReply) => {
    const { password } = request.body;
    const hash = await hashPassword(password);
    return reply.code(200).send(hash);
  });
  */
  
  //------------------------------------------------------------------
  
  done();
}

//------------------------------------------------------------------
