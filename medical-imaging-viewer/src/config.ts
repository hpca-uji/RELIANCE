import path from "path";
import dotenv from "dotenv";

// Parsing the env file.
dotenv.config({ path: path.resolve(__dirname, "../config/config.env") });

// Interface to load env variables
// Note these variables can possibly be undefined
// as someone could skip these varibales or not setup a .env file at all

interface ENV {
    DB_PORT: number | undefined;
    DB_HOST: string | undefined;
    DB_USER: string | undefined;
    DB_PWD: string | undefined;
    DB_NAME: string | undefined;
    JWT_TOKEN: string | undefined;
    REFRESH_SESSION_NAME: string | undefined;
    REFRESH_COOKIE_NAME: string | undefined;
    SESSION_NAME: string | undefined;
    COOKIE_NAME: string | undefined;
}

interface Config {
    DB_PORT: number;
    DB_HOST: string;
    DB_USER: string;
    DB_PWD: string;
    DB_NAME: string;
    JWT_TOKEN: string;
    REFRESH_SESSION_NAME: string;
    REFRESH_COOKIE_NAME: string;
    SESSION_NAME: string;
    COOKIE_NAME: string;
}

// Loading process.env as ENV interface

const getConfig = (): ENV => {
  return {
    DB_PORT: process.env.DB_PORT ? Number(process.env.DB_PORT) : undefined,
    DB_HOST: process.env.DB_HOST,
    DB_USER: process.env.DB_USER,
    DB_PWD: process.env.DB_PWD,
    DB_NAME: process.env.DB_NAME,
    JWT_TOKEN: process.env.JWT_TOKEN,
    REFRESH_SESSION_NAME: process.env.REFRESH_SESSION_NAME,
    REFRESH_COOKIE_NAME: process.env.REFRESH_COOKIE_NAME,
    SESSION_NAME: process.env.SESSION_NAME,
    COOKIE_NAME: process.env.COOKIE_NAME
  };
};

// Throwing an Error if any field was undefined we don't 
// want our app to run if it can't connect to DB and ensure 
// that these fields are accessible. If all is good return
// it as Config which just removes the undefined from our type 
// definition.

const getSanitzedConfig = (config: ENV): Config => {
  for (const [key, value] of Object.entries(config)) {
    if (value === undefined) {
      throw new Error(`Missing key ${key} in config.env`);
    }
  }
  return config as Config;
};

const config = getConfig();

const sanitizedConfig = getSanitzedConfig(config);

export default sanitizedConfig;