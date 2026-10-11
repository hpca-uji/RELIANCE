import { promises } from 'fs';
import path from 'path';
import { ConfParams, config } from './config';
import { FastifyBaseLogger } from 'fastify';


const getDirectories = async (source: string, log: FastifyBaseLogger) => {
  try {
    const dir = await promises.readdir(source, { withFileTypes: true })
    return dir.filter((dirent) => dirent.isDirectory()).map((dirent) => dirent.name)
  }
  catch (e) {
    log.warn("Storage Folder doesn't exist: ", source);
    return [];
  }
}

export async function fileExists(pathname: string): Promise<boolean> {
  try {
    const stat = await promises.stat(pathname);
    return !!stat;
  }
  catch {
    return false;
  }
}

export async function clearCache(log: FastifyBaseLogger) {
  const storagePath = config.get(ConfParams.STORAGE_PATH) as string;
  const retention = config.get(ConfParams.CACHE_RETENTION) as number;

  if (retention < 0) {
    log.info('cache cleanup disabled');
    return;
  }

  const dirs = await getDirectories(storagePath, log);
  const dateNow = new Date();

  for (const dir of dirs) {
    const filepath = path.join(storagePath, dir);
    const stats = await promises.stat(filepath);
    const mtime = stats.mtime;
    const minutes = (dateNow.getTime() - mtime.getTime()) / 60000;
    if (minutes > retention) {
      log.info(`removing: ${filepath}`);
      await promises.rm(filepath, { recursive: true, force: true });
    }
  }
}
