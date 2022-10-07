-- PostgreSQL

CREATE USER dbuser WITH PASSWORD 'topsecret';
CREATE DATABASE adult OWNER dbuser;
CREATE TABLE adult(id SERIAL primary key, age INTEGER, workclass VARCHAR (255), fnlwgt INTEGER, education VARCHAR (255), educationalnum INTEGER, maritalstatus VARCHAR (255), occupation VARCHAR (255), relationship VARCHAR (255), race VARCHAR (255), gender VARCHAR (255), capitalgain INTEGER, capitalloss INTEGER, hoursperweek INTEGER, nativecountry VARCHAR (255), income VARCHAR (255));
COPY adult(age, workclass, fnlwgt, education, educationalnum, maritalstatus, occupation, relationship, race, gender, capitalgain, capitalloss, hoursperweek, nativecountry, income) FROM '/home/jens/master-thesis/data/adult.csv' DELIMITER ',' CSV HEADER;

-- MySql, outdated. Use PostgreSQL

CREATE DATABASE adult;
CREATE TABLE adult(id INT(255) primary key auto_increment, age INT(255), workclass VARCHAR(255), fnlwgt INT(255), education VARCHAR(255), educationalnum INT(255), maritalstatus VARCHAR(255), occupation VARCHAR(255), relationship VARCHAR(255), race VARCHAR(255), gender VARCHAR(255), capitalgain INT(255), capitalloss INT(255), hoursperweek INT(255), nativecountry VARCHAR(255), income VARCHAR(255));
LOAD DATA INFILE '/var/lib/mysql-files/adult.csv' INTO TABLE adult FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS (age, workclass, fnlwgt, education, educationalnum, maritalstatus, occupation, relationship, race, gender, capitalgain, capitalloss, hoursperweek, nativecountry, income);

