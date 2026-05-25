# Creating Database:
create database ecommerce_db;

# Creating User:
## SECURITY NOTE: Replace <INSERT_SECURE_PASSWORD_HERE> with a strong password before executing this in MySQL Workbench.
## DO NOT commit the real password.
create user 'agentic_app'@'localhost' identified by '<INSERT_SECURE_PASSWORD_HERE>';

# Checking DB and User Creation:
show databases;
select user, host from mysql.user;

# Granting all permissions on ecommerce_db to New User:
grant all privileges on ecommerce_db.* to 'agentic_app'@'localhost';

# Reloading the Privileges Table:
flush privileges;