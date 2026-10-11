import { ConfParams, config } from '../utils/config';
import { recompress, recompressOptions } from 'reliance-dicom-dimse-native';
import { FastifyBaseLogger } from 'fastify/types/logger';

export async function compressFile(inputFile: string, outputDirectory: string, log: FastifyBaseLogger, transferSyntax: string | undefined = undefined) {
  const options: recompressOptions = {
    sourcePath: inputFile,
    storagePath: outputDirectory,
    writeTransfer: transferSyntax || config.get(ConfParams.XTRANSFER),
    lossyQuality: config.get(ConfParams.LOSSY_QUALITY),
    verbose: config.get(ConfParams.VERBOSE),
  };

  // run find scu and return json response
  return new Promise((resolve, reject) => {
    recompress(options, (result: string) => {
      if (result && result.length > 0) {
        try {
          const json = JSON.parse(result);
          if (json.code === 0) {
            resolve(true);
          } else {
            log.error(`recompression failure (${inputFile}): ${json.message}`);
            reject();
          }
        } catch (error) {
          log.error(error);
          log.error(result);
          reject();
        }
      } else {
        log.error('invalid result received');
        reject();
      }
    });
  });
}
