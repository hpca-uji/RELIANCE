declare namespace NodeJS {
    interface ProcessEnv {
        DB_PORT: string;
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
}
