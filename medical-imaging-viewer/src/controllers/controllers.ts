
import { FastifyRequest, FastifyReply } from 'fastify';

import { validatePassword } from '../utils/hash';
import { IUsers, diseaseIds } from '../db/tables';
import { ContactsRepository, UsersRepository, DicomsRepository, RadiologistPredictionsRepository } from '../db/models';
import { IDataBaseConfiguration } from '../utils/configurationInterfaces';

import config from "../config";
import extractIdFromRequest from '../utils/decoder';

function getDiseaseIdFromLabel(label: string) : number | undefined{
  return diseaseIds.get(label);
}

export const postLogin = async (req: FastifyRequest<{ Body: {username: string, password: string} }>, reply: FastifyReply) => {
    const { username, password } = req.body;

    try {
        // getHash from username from DB
        //const databaseConfiguration: IDataBaseConfiguration = { host: config.DB_HOST, port: config.DB_PORT, user: config.DB_USER, password: config.DB_PWD, database: config.DB_NAME};
        const user: IUsers = await UsersRepository.readByUser(req.server, username);

        if (!user) {
          req.log.warn('Username not found ' + username);
          return reply.code(401).view('login.ejs', { error: 'Invalid credentials.' });
        }

        const isValid = await validatePassword(password, user.Password);
        if (!isValid){
          req.log.warn('Not valid password for username ' + username);
          return reply.code(401).view('login.ejs', { error: 'Invalid credentials.' });
        }
        reply.jwtSign({userId: user.UserID, username: username, isAdmin: user.IsAdmin}).then(async (token) => {
          req.appToken.set('data', token)
          req.appRefresh.set('data', token)
          return await reply.redirect('/');
        });
    } catch (error) {
      req.log.error(error);
      reply.send(500);
      return;
    }
}

export const postContact = async (req: FastifyRequest<{ Body: {username: string, email: string, subject: string} }>, reply: FastifyReply) => {
  const { username, email, subject } = req.body;

  try {
      // Insert in DB
      //const databaseConfiguration: IDataBaseConfiguration = { host: config.DB_HOST, port: config.DB_PORT, user: config.DB_USER, password: config.DB_PWD, database: config.DB_NAME};
      const { isCreated, err} = await ContactsRepository.create(req.server, username, email, subject);

      if (!isCreated){
        req.log.error(err);
        return reply.code(401).view('contact.ejs', { error: 'Unable to process the request.' });
      }

      return reply.code(200).view('contact.ejs', { msg: 'Message received. We will contact you as soon as possible.' });
  } catch (error) {
    req.log.error(error);
    reply.send(500);
    return;
  }
}


export const postLabels = async (req: FastifyRequest<{ Body: {checkedLabels: string[], studyInstanceUID: string, seriesInstanceUID: string, sopInstanceUID: string} }>, reply: FastifyReply) => {
  const { checkedLabels, studyInstanceUID, seriesInstanceUID, sopInstanceUID } = req.body;

  try {
    const userId = extractIdFromRequest(req);
    if (userId === undefined) return reply.send(500); 
    req.log.info('/labels: ' + JSON.stringify(userId, null, 2));
    //const databaseConfiguration: IDataBaseConfiguration = { host: config.DB_HOST, port: config.DB_PORT, user: config.DB_USER, password: config.DB_PWD, database: config.DB_NAME};
    
    // getDicomId from DB: studyInstanceUID --> accessionnumber
    const dicomId: string | undefined = await DicomsRepository.readId(req.server, studyInstanceUID, seriesInstanceUID, sopInstanceUID);
    if (dicomId === undefined) return reply.send(500);

    // parse labels from web page
    let labelsIds = Object.values(checkedLabels).map((value: string): number | undefined => {
      return getDiseaseIdFromLabel(value);
    });
    // delete undefined
    labelsIds = labelsIds.filter(function( element ) {
      return element !== undefined;
    });

    // insert diseaseId and DB insertLabel(label, studyInstanceUID, userId)
    const allInserted: boolean = await RadiologistPredictionsRepository.insertPrediction(req.server, userId, dicomId, labelsIds);

    return reply.send(200);
  } catch (error) {
    console.log(error);
    return reply.send(500);
  }
}

export const postDiscard = async (req: FastifyRequest<{ Body: {studyInstanceUID: string, seriesInstanceUID: string, sopInstanceUID: string} }>, reply: FastifyReply) => {
  const { studyInstanceUID, seriesInstanceUID, sopInstanceUID } = req.body;

  try {
    const userId = extractIdFromRequest(req);
    if (userId === undefined) return reply.send(500); 
    req.log.info('discard: ' + JSON.stringify(userId, null, 2));
    //const databaseConfiguration: IDataBaseConfiguration = { host: config.DB_HOST, port: config.DB_PORT, user: config.DB_USER, password: config.DB_PWD, database: config.DB_NAME};
    
    // getDicomId from DB: studyInstanceUID --> accessionnumber
    const dicomId: string | undefined = await DicomsRepository.readId(req.server, studyInstanceUID, seriesInstanceUID, sopInstanceUID);
    if (dicomId === undefined) return reply.send(500);

    // mark dicom as discarded in DB
    const updated: boolean = await DicomsRepository.discard(req.server, dicomId, userId);

    return reply.send(200);
  } catch (error) {
    console.log(error);
    return reply.send(500);
  }
}
