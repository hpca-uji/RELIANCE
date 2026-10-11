import { fetchMeta } from '../dimse/fetchMeta';
import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { doFind } from '../dimse/findData';
import { QUERY_LEVEL } from '../dimse/querLevel';
import { doWadoRs } from '../dimse/wadoRs';
import { doWadoUri } from '../dimse/wadoUri';

import deepmerge from 'deepmerge';
import combineMerge from '../utils/combineMerge';
import extractIdFromRequest from '../utils/decoder';
import config from '../config';

const options = { arrayMerge: combineMerge };

interface IParamsStudy {
  studyInstanceUid: string;
}

interface IParamsSeries extends IParamsStudy {
  seriesInstanceUid: string;
}

interface IParamsImage extends IParamsSeries {
  sopInstanceUid: string;
}

interface IQueryImage {
  studyUID: string;
  seriesUID: string;
  objectUID: string;
}

interface QueryParams {
  [key: string]: string;
}

module.exports = function (server: FastifyInstance, opts: unknown, done: () => void) {
  server.addHook("onRequest", async (request: FastifyRequest, reply: FastifyReply) => {
    try {
      request.log.info("ohif check token");
      const token = request.appToken.get('data') || '';
      server.jwt.verify(token);
      request.log.info("OHIF session token is valid");
    } catch (err) {
      try{
        request.log.info("OHIF session token is invalid");
        const refreshToken = request.appRefresh.get('data') || '';
        server.jwt.verify(refreshToken);
        request.log.info("OHIF refresh token is invalid");
        request.appToken.set('data', refreshToken);
        request.appRefresh.set('data', refreshToken);
      } catch(err2){
        request.log.warn("ohif Authorization not found in request.");
        return reply.view('login.ejs', { error: 'Sign in to access this resource.' });
      }
    }
  });

  //------------------------------------------------------------------

  server.get<{
    Querystring: QueryParams;
  }>(
    '/rs/studies',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      try {
        const { query } = req;
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies: ' + JSON.stringify(userId, null, 2));
        const json = deepmerge.all(await doFind(req.server, query, userId), options);
        return reply.send(json);
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsStudy;
    Querystring: QueryParams;
  }>(
    '/rs/studies/:studyInstanceUid',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { params } = req;
      const { studyInstanceUid } = params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer); 
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsStudy;
  }>(
    '/rs/studies/:studyInstanceUid/pixeldata',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid } = req.params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/pixeldata: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, dataFormat: 'pixeldata' }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      }
      catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsStudy;
  }>(
    '/rs/studies/:studyInstanceUid/rendered',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid } = req.params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500);
        req.log.info('/rs/studies/:studyInstanceUid/rendered: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, dataFormat: 'rendered' }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      }
      catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsStudy;
  }>(
    '/rs/studies/:studyInstanceUid/thumbnail',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid } = req.params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/thumbnail: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, dataFormat: 'thumbnail' }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      }
      catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsStudy;
    Querystring: QueryParams;
  }>(
    '/rs/studies/:studyInstanceUid/metadata',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { query } = req;
      query.StudyInstanceUID = req.params.studyInstanceUid;

      try {
        const { query } = req;
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/metadata: ' + JSON.stringify(userId, null, 2));
        const json = deepmerge.all(await doFind(req.server, query, userId), options);
        return reply.send(json);
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsStudy;
    Querystring: QueryParams;
  }>(
    '/rs/studies/:studyInstanceUid/series',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { query } = req;
      query.StudyInstanceUID = req.params.studyInstanceUid;

      try {
        const { query } = req;
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/series: ' + JSON.stringify(userId, null, 2));
        const json = deepmerge.all(await doFind(req.server, query, userId), options);
        return reply.send(json);
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsSeries;
    Querystring: QueryParams;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { params } = req;
      const { studyInstanceUid, seriesInstanceUid } = params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, seriesInstanceUid }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });
  
  //------------------------------------------------------------------

  server.get<{
    Params: IParamsSeries;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/pixeldata',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid, seriesInstanceUid } = req.params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/pixeldata: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, seriesInstanceUid, dataFormat: 'pixeldata' }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      }
      catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsSeries;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/rendered',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid, seriesInstanceUid } = req.params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/rendered: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, seriesInstanceUid, dataFormat: 'rendered' }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      }
      catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsSeries;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/thumbnail',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid, seriesInstanceUid } = req.params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/thumbnail: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, seriesInstanceUid, dataFormat: 'thumbnail' }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      }
      catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsSeries;
    Querystring: QueryParams;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { query, params } = req;
      query.StudyInstanceUID = params.studyInstanceUid;
      query.SeriesInstanceUID = params.seriesInstanceUid;

      try {
        const { query } = req;
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances token: ' + JSON.stringify(userId, null, 2));
        const json = deepmerge.all(await doFind(req.server, query, userId), options);
        return reply.send(json);
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------
  server.get<{
    Params: IParamsSeries;
    Querystring: QueryParams;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/metadata',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid, seriesInstanceUid } = req.params;
      const { query } = req;
      query.StudyInstanceUID = studyInstanceUid;
      query.SeriesInstanceUID = seriesInstanceUid;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/metadata: ' + JSON.stringify(userId, null, 2));
        const rsp = await fetchMeta(req.server, query, studyInstanceUid, seriesInstanceUid, userId, req.log);
        return reply.send(rsp);
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsImage;
    Querystring: QueryParams;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid/metadata',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid, seriesInstanceUid, sopInstanceUid } = req.params;
      const { query } = req;
      query.StudyInstanceUID = studyInstanceUid;
      query.SeriesInstanceUID = seriesInstanceUid;
      query.SopnstanceUID = sopInstanceUid;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid/metadata: ' + JSON.stringify(userId, null, 2));
        const rsp = await fetchMeta(req.server, query, studyInstanceUid, seriesInstanceUid, userId, req.log);
        return reply.send(rsp);
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsImage;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid/pixeldata',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid, seriesInstanceUid, sopInstanceUid } = req.params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid/pixeldata: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, seriesInstanceUid, sopInstanceUid, dataFormat: 'pixeldata' }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      }
      catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsImage;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid/rendered',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid, seriesInstanceUid, sopInstanceUid } = req.params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500); 
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid/rendered: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, seriesInstanceUid, sopInstanceUid, dataFormat: 'rendered' }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      }
      catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsImage;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid/thumbnail',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid, seriesInstanceUid, sopInstanceUid } = req.params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500);
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid/thumbnail: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, seriesInstanceUid, sopInstanceUid, dataFormat: 'thumbnail' }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      }
      catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsImage;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid/frames/:frame',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyInstanceUid, seriesInstanceUid, sopInstanceUid } = req.params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500);
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid/frames/:frame: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, seriesInstanceUid, sopInstanceUid, dataFormat: 'pixeldata' }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Params: IParamsImage;
    Querystring: QueryParams;
  }>(
    '/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { params } = req;
      const { studyInstanceUid, seriesInstanceUid, sopInstanceUid } = params;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500);
        req.log.info('/rs/studies/:studyInstanceUid/series/:seriesInstanceUid/instances/:sopInstanceUid: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoRs(req.server, { studyInstanceUid, seriesInstanceUid, sopInstanceUid }, userId, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  server.get<{
    Querystring: IQueryImage;
  }>(
    '/wadouri',
    //{ onRequest: [server.verifyJWT] },
    async (req, reply) => {
      const { studyUID, seriesUID, objectUID } = req.query;

      try {
        const userId = extractIdFromRequest(req);
        if (userId === undefined) return reply.send(500);
        req.log.info('/wadouri: ' + JSON.stringify(userId, null, 2));
        const rsp = await doWadoUri({ studyInstanceUid: studyUID, seriesInstanceUid: seriesUID, sopInstanceUid: objectUID }, req.log);
        reply.header('Content-Type', rsp.contentType);
        return reply.send(rsp.buffer);
      } catch (error) {
        req.log.error(error);
        return reply.send(500);
      }
  });

  //------------------------------------------------------------------

  done();
}

//------------------------------------------------------------------
