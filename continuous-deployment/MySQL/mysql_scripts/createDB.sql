#DROP DATABASE IF EXISTS app_db;

#CREATE DATABASE app_db;
#USE app_db;

CREATE TABLE ChestPredictions(
	id varchar(255) PRIMARY KEY,
	predictions text NOT NULL,
	acquisitiontime datetime,
	receivedtime datetime NOT NULL,
	processedtime datetime NOT NULL
);

#SHOW DATABASES;
#DESCRIBE app_db.ChestPredictions;
