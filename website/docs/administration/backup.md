---
sidebar_position: 2
---

# Data Backup

:::info
We recommend regularly backing up the database to ensure data recovery in case of a failure. It's particularly advisable to back up before making significant changes such as upgrades or configuration modifications.
:::

By default, Tabby stores all its data in the `$HOME/.tabby` directory. However, you can override this behavior by using the `TABBY_ROOT` environment variable. This directory contains all the necessary data for Tabby's operation, including the database, logs, and configuration files.

## Database Backup

Tabby uses SQLite for data storage. You can back up the database by simply copying the file to a secure location. The database is located at `$HOME/.tabby/ee/db.sqlite` by default.

## Event Logs Backup

Tabby stores all event logs in the `~/.tabby/events` directory. These events, stored in JSON format, are named after the date of their creation.

```
% ls ~/.tabby/events

2023-11-24.json 2023-12-08.json 2024-01-09.json 2024-01-31.json 2024-02-10.json 2024-02-22.json 2024-03-06.json
2023-11-26.json 2023-12-09.json 2024-01-17.json 2024-02-01.json 2024-02-11.json 2024-02-23.json 2024-03-07.json
2023-11-28.json 2023-12-10.json 2024-01-18.json 2024-02-02.json 2024-02-12.json 2024-02-24.json 2024-03-10.json
2023-11-29.json 2023-12-11.json 2024-01-19.json 2024-02-03.json 2024-02-13.json 2024-02-25.json 2024-03-13.json
2023-11-30.json 2023-12-15.json 2024-01-21.json 2024-02-04.json 2024-02-14.json 2024-02-26.json 2024-03-20.json
2023-12-01.json 2023-12-16.json 2024-01-22.json 2024-02-05.json 2024-02-15.json 2024-02-27.json
2023-12-02.json 2023-12-18.json 2024-01-23.json 2024-02-06.json 2024-02-16.json 2024-03-01.json
2023-12-04.json 2023-12-19.json 2024-01-26.json 2024-02-07.json 2024-02-18.json 2024-03-02.json
2023-12-05.json 2023-12-20.json 2024-01-27.json 2024-02-08.json 2024-02-19.json 2024-03-03.json
2023-12-07.json 2023-12-22.json 2024-01-30.json 2024-02-09.json 2024-02-20.json 2024-03-05.json
```