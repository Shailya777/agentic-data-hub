# Creating Database:
create database ecommerce_db;

# Creating User:
create user 'agentic_app'@'localhost' identified by '######';

# Checking DB and User Creation:
show databases;
select user, host from mysql.user;

# Granting all permissions on ecommerce_db to New User:
grant all privileges on ecommerce_db.* to 'agentic_app'@'localhost';

# Reloading the Privileges Table:
flush privileges;