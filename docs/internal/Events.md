# Events System

In Tabby, we use the `vector` tool to collect logs from various sources, transform them into a standard `Event`, and persist them in `/data/logs/events`.

## Schema

### Event

```jsx
{
  "id": EventId,
  "data": Any
}
```

The `id` field can be used to uniquely identify an event.

The `data`  field is a standard JSON object, and its definition is left to downstream tasks.

### EventId

```jsx
{
  "process_id": Number,
  "thread_id": Number,
  // Unix timestamp
  "timestamp": Number,
}
```

In the future, we might add `server_id` when Tabby evolves into a distributed environment.
