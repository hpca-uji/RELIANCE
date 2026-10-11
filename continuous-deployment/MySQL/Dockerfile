FROM mysql:8

ENV MYSQL_DATABASE app_db

COPY ./mysql_scripts/ /docker-entrypoint-initdb.d/
