# Master Thesis: Semi-Supervised QBE System

## About

This is a Query by Example system developed for the master thesis "Semi-Supervised machine learning for Query by Example on relational databases". It implements three different algorithms, decision trees, random forest and gradient boosting, each as supervised and semi-supervised version. These models can be hyper-parameter tuned on a dataset (census dataset "adult") and then used for predicting the query output of a user-intended query based on the provided examples. 
Additionally, a benchmark has been conducted for the thesis, which can also be found in this repository.

## Installation

- Clone the repository and cd into it
- Install the requirements from requirements.txt for the webapp, benchmark or both. For example:
```pip install -r webapp/requirements.txt```
- Set up a postgres database on localhost
- Create the user 'dbuser' with password 'topsecret' (or change the user parameters later on in the settings.py of 'webapp')
- Create two databases: ```qbe_app``` and ```adult``` both owned by dbuser
- Import the two database dumps: <br>
  ```psql qbe_app < webapp/data/db_dumps/qbe_system_db -U dbuser -h localhost``` and <br>
  ```psql adult < webapp/data/db_dumps/adult_db -U dbuser -h localhost```
- If needed, change the database connection parameters in the settings.py of 'webapp'
- Start the system from the 'webapp' folder by running
  ```python3 manage.py runserver```

##  Helpful Commands
 - Starting Postgres in WSL Ubuntu:
    ```sudo service postgresql start```
 - Starting Postgres in normal Ubuntu:
    ```sudo systemctl start postgresql.service```

## Known Issues
 - Image of tree visualization on "Run Query" page not refreshing:
    - Delete the browser cache and refresh the page


