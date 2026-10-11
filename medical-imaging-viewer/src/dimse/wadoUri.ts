import { ConfParams, config } from '../utils/config';
import { fileExists } from '../utils/fileHelper';
import { compressFile } from './compressFile';
import { waitOrFetchData } from './fetchData';
import path from 'path';
import fs from 'fs';
import { stringToQueryLevel } from './querLevel';
import { FastifyBaseLogger } from 'fastify/types/logger';

type WadoUriArgs = {
  studyInstanceUid: string;
  seriesInstanceUid: string;
  sopInstanceUid: string;
};
type WadoUriResponse = {
  contentType: string;
  buffer: Buffer;
};
export async function doWadoUri({ studyInstanceUid, seriesInstanceUid, sopInstanceUid }: WadoUriArgs, log: FastifyBaseLogger): Promise<WadoUriResponse> {
  const fetchLevel = config.get(ConfParams.FETCH_LEVEL) as string;
  const level = stringToQueryLevel(fetchLevel);

  const storagePath = config.get(ConfParams.STORAGE_PATH) as string;
  const studyPath = path.join(storagePath, studyInstanceUid);
  const pathname = path.join(studyPath, sopInstanceUid);
  
  // fetch if needed
  const exists = await fileExists(pathname);
  if (!exists) {
    try {
      await waitOrFetchData(studyInstanceUid, seriesInstanceUid, sopInstanceUid, level, log);
    } catch (err) {
      log.error(`fetch failed for study: ${studyInstanceUid}`);
      throw err;
    }
  }

  const postExists = await fileExists(pathname);
  if (!postExists) {
    const msg = `file not found ${pathname}`;
    log.error(msg);
    throw msg;
  }

  try {
    await compressFile(pathname, studyPath, log);
  } catch (error) {
    log.error(error);
    const msg = `failed to compress ${pathname}`;
    throw msg;
  }

  // read file from file system
  const fsPromise = fs.promises;
  try {
    return {
      contentType: 'application/dicom',
      buffer: await fsPromise.readFile(pathname),
    };
  } catch (error) {
    log.error(error);
    const msg = `failed to read ${pathname}`;
    throw msg;
  }
}
