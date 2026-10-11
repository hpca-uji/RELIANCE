import { sendCFindRequest } from './findData';
import { config, ConfParams } from '../utils/config';
import { fileExists } from '../utils/fileHelper';
import { QUERY_LEVEL } from './querLevel';
import { waitOrFetchDataOnAet } from './fetchData';
import { parseMeta } from './parseMeta';
import { IConfigurationPeers } from '../utils/configurationInterfaces';

import path from 'path';
import { FastifyInstance } from 'fastify';
import { FastifyBaseLogger } from 'fastify/types/logger';

export async function fetchMeta(server: FastifyInstance, query: any, studyInstanceUID: string, seriesInstanceUID: string, userId: string, log: FastifyBaseLogger): Promise<unknown> {
  const peers = config.get(ConfParams.PEERS) as IConfigurationPeers[];
  const fullMeta = config.get(ConfParams.FULL_META) as boolean;

  for(const {db, ...peer} of peers) {
    log.info(`checking database for data: ${db.host}`);
    const json = (await sendCFindRequest(server, db, query, userId)) as any;
    // make sure c-find worked
    if (json.length === 0) {
      log.info(`no data found on peer: ${db.host}`);
      continue;
    }

    if (!fullMeta) {
      return json;
    }

    // check if fetch is needed
    for(const image of json) {
      const sopInstanceUid = image['00080018'].Value[0] as string;
      const storagePath = config.get(ConfParams.STORAGE_PATH) as string;
      const pathname = path.join(storagePath, studyInstanceUID, sopInstanceUid);
      const exists = await fileExists(pathname);
      if (!exists) {
        log.info(`fetching image ${sopInstanceUid}`);
        await waitOrFetchDataOnAet(studyInstanceUID, seriesInstanceUID, sopInstanceUid, QUERY_LEVEL.IMAGE, peer, log);
        break;
      }
    }
    try {
      const result = await parseMeta(json, studyInstanceUID, seriesInstanceUID, log);
      log.info('parsing finished, resolving');
      return Promise.resolve(result);
    } catch (error) {
      log.error(error);
      throw error;
    }
  }
  return Promise.reject('failed fetching meta data');
}
