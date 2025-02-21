use anyhow::{bail, Result};
use chrono::{DateTime, Duration, Utc};
use sqlx::{query, query_as, types::Json, FromRow};
use tabby_db_macros::query_paged_as;

use crate::{
    attachment::{
        Attachment, AttachmentClientCode, AttachmentCode, AttachmentCodeFileList, AttachmentDoc,
    },
    AsSqliteDateTimeString, DbConn,
};

#[derive(FromRow)]
pub struct ThreadDAO {
    pub id: i64,
    pub user_id: i64,
    pub is_ephemeral: bool,
    pub relevant_questions: Option<Json<Vec<String>>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(sqlx::FromRow)]
pub struct ThreadMessageDAO {
    pub id: i64,
    pub thread_id: i64,

    pub role: String,
    pub content: String,

    pub code_source_id: Option<String>,
    pub attachment: Option<Json<Attachment>>,

    // Deprecated since 0.25 (not removed from db yet).
    // FIXME(meng): remove these columns from db in 0.26.
    // pub code_attachments: Option<Json<Vec<AttachmentCode>>>,
    // pub client_code_attachments: Option<Json<Vec<AttachmentClientCode>>>,
    // pub doc_attachments: Option<Json<Vec<AttachmentDoc>>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl DbConn {
    pub async fn create_thread(&self, user_id: i64, is_ephemeral: bool) -> Result<i64> {
        let res = query!(
            "INSERT INTO threads(user_id, is_ephemeral) VALUES (?, ?)",
            user_id,
            is_ephemeral
        )
        .execute(&self.pool)
        .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn list_threads(
        &self,
        ids: Option<&[i64]>,
        user_id: Option<i64>,
        is_ephemeral: Option<bool>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<ThreadDAO>> {
        let mut conditions = vec![];

        if let Some(ids) = ids {
            let ids: Vec<String> = ids.iter().map(i64::to_string).collect();
            let ids = ids.join(", ");
            conditions.push(format!("id in ({ids})"));
        }

        if let Some(user_id) = user_id {
            conditions.push(format!("user_id = {user_id}"));
        }

        if let Some(is_ephemeral) = is_ephemeral {
            if is_ephemeral {
                conditions.push("is_ephemeral".to_string());
            } else {
                conditions.push("NOT is_ephemeral".to_string());
            }
        }

        let condition = (!conditions.is_empty()).then_some(conditions.join(" AND "));
        let threads = query_paged_as!(
            ThreadDAO,
            "threads",
            [
                "id",
                "user_id",
                "is_ephemeral",
                "relevant_questions" as "relevant_questions: Json<Vec<String>>",
                "created_at" as "created_at: DateTime<Utc>",
                "updated_at" as "updated_at: DateTime<Utc>"
            ],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(threads)
    }

    pub async fn update_thread_relevant_questions(
        &self,
        thread_id: i64,
        relevant_questions: &[String],
    ) -> Result<()> {
        let relevant_questions = Json(relevant_questions.to_vec());
        query!(
            "UPDATE threads SET relevant_questions = ?, updated_at = DATETIME('now') WHERE id = ?",
            relevant_questions,
            thread_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_thread_ephemeral(&self, thread_id: i64, is_ephemeral: bool) -> Result<()> {
        query!(
            "UPDATE threads SET is_ephemeral = ?, updated_at = DATETIME('now') WHERE id = ?",
            is_ephemeral,
            thread_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn create_thread_message(
        &self,
        thread_id: i64,
        role: &str,
        content: &str,
        code_attachments: Option<&[AttachmentCode]>,
        client_code_attachments: Option<&[AttachmentClientCode]>,
        doc_attachments: Option<&[AttachmentDoc]>,
        verify_last_message_role: bool,
    ) -> Result<i64> {
        if verify_last_message_role {
            let last_message = self.get_last_thread_message(thread_id).await?;
            if let Some(last_message) = last_message {
                if last_message.role == role {
                    bail!("Cannot send two messages in a row with the same role");
                }
            }
        }

        let code_attachments = code_attachments.map(Json);
        let client_code_attachments = client_code_attachments.map(Json);
        let doc_attachments = doc_attachments.map(Json);
        let res = query!(
            r#"INSERT INTO thread_messages(
                thread_id,
                role,
                content,
                attachment
            ) VALUES (?, ?, ?, JSON_OBJECT('code', JSON(?), 'client_code', JSON(?), 'doc', JSON(?)))"#,
            thread_id,
            role,
            content,
            code_attachments,
            client_code_attachments,
            doc_attachments,
        )
        .execute(&self.pool)
        .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn update_thread_message_code_file_list_attachment(
        &self,
        message_id: i64,
        file_list: &[String],
    ) -> Result<()> {
        let code_file_list_attachment = Json(AttachmentCodeFileList {
            file_list: file_list.into(),
        });
        query!(
            "UPDATE thread_messages SET attachment = JSON_SET(attachment, '$.code_file_list', JSON(?)), updated_at = DATETIME('now') WHERE id = ?",
            code_file_list_attachment,
            message_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_thread_message_code_attachments(
        &self,
        message_id: i64,
        code_attachments: &[AttachmentCode],
    ) -> Result<()> {
        let code_attachments = Json(code_attachments);
        query!(
            "UPDATE thread_messages SET attachment = JSON_SET(attachment, '$.code', JSON(?)), updated_at = DATETIME('now') WHERE id = ?",
            code_attachments,
            message_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_thread_message_code_source_id(
        &self,
        message_id: i64,
        code_source_id: &str,
    ) -> Result<()> {
        query!(
            "UPDATE thread_messages SET code_source_id = ?, updated_at = DATETIME('now') WHERE id = ?",
            code_source_id,
            message_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_thread_message_doc_attachments(
        &self,
        message_id: i64,
        doc_attachments: &[AttachmentDoc],
    ) -> Result<()> {
        let doc_attachments = Json(doc_attachments);
        query!(
            "UPDATE thread_messages SET attachment = JSON_SET(attachment, '$.doc', JSON(?)), updated_at = DATETIME('now') WHERE id = ?",
            doc_attachments,
            message_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_thread_message_content(
        &self,
        thread_id: i64,
        message_id: i64,
        content: &str,
    ) -> Result<()> {
        query!(
            "UPDATE thread_messages SET content = ?, updated_at = DATETIME('now') WHERE thread_id = ? AND id = ?",
            content,
            thread_id,
            message_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn append_thread_message_content(
        &self,
        message_id: i64,
        content: &str,
    ) -> Result<()> {
        query!(
            "UPDATE thread_messages SET content = content || ?, updated_at = DATETIME('now') WHERE id = ?",
            content,
            message_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn get_last_thread_message(&self, thread_id: i64) -> Result<Option<ThreadMessageDAO>> {
        let message = query_as!(
            ThreadMessageDAO,
            r#"SELECT
                id,
                thread_id,
                role,
                content,
                code_source_id,
                attachment as "attachment: Json<Attachment>",
                created_at as "created_at: DateTime<Utc>",
                updated_at as "updated_at: DateTime<Utc>"
            FROM thread_messages
            WHERE thread_id = ?
            ORDER BY id DESC
            LIMIT 1"#,
            thread_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(message)
    }

    pub async fn list_thread_messages(
        &self,
        thread_id: i64,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<ThreadMessageDAO>> {
        let condition = format!("thread_id = {}", thread_id);
        let messages = query_paged_as!(
            ThreadMessageDAO,
            "thread_messages",
            [
                "id",
                "thread_id",
                "role",
                "content",
                "code_source_id",
                "attachment" as "attachment: Json<Attachment>",
                "created_at" as "created_at: DateTime<Utc>",
                "updated_at" as "updated_at: DateTime<Utc>"
            ],
            limit,
            skip_id,
            backwards,
            Some(condition)
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(messages)
    }

    pub async fn delete_thread_message_pair(
        &self,
        thread_id: i64,
        user_message_id: i64,
        assistant_message_id: i64,
    ) -> Result<()> {
        #[derive(FromRow)]
        struct Response {
            id: i64,
            role: String,
        }

        let message = query_as!(
            Response,
            "SELECT id, role FROM thread_messages WHERE thread_id = ? AND id >= ? AND id <= ?",
            thread_id,
            user_message_id,
            assistant_message_id
        )
        .fetch_all(&self.pool)
        .await?;

        if message.len() != 2 {
            bail!("Thread message pair is not valid")
        }

        let is_valid_user_message = message[0].id == user_message_id && message[0].role == "user";
        let is_valid_assistant_message =
            message[1].id == assistant_message_id && message[1].role == "assistant";

        if !is_valid_user_message || !is_valid_assistant_message {
            bail!("Invalid message pair");
        }

        query!(
            "DELETE FROM thread_messages WHERE id = ? or id = ?",
            user_message_id,
            assistant_message_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn delete_thread(&self, id: i64) -> Result<()> {
        query!("DELETE FROM threads WHERE id = ?", id,)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn delete_expired_ephemeral_threads(&self) -> Result<usize> {
        let time = (Utc::now() - Duration::days(7)).as_sqlite_datetime();

        let res = query!(
            r#"
            DELETE FROM threads
            WHERE
                id IN (
                    SELECT
                        id
                    FROM
                        (
                            SELECT
                                threads.id,

                                --- Get the latest updated_at time from the thread and its messages using MAX aggregation
                                MAX(
                                    CASE
                                        --- If there are no messages, use the thread's updated_at time
                                        WHEN thread_messages.updated_at IS NULL THEN threads.updated_at

                                        --- Otherwise, use the message's updated_at time
                                        ELSE thread_messages.updated_at
                                    END
                                ) AS last_updated_at
                            FROM
                                threads
                                LEFT JOIN thread_messages ON threads.id = thread_messages.thread_id
                            WHERE
                                is_ephemeral
                            GROUP BY
                                1
                            HAVING
                                last_updated_at < ?
                        )
                );
        "#,
            time
        )
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() as usize)
    }
}

#[cfg(test)]
mod tests {
    use crate::{testutils, DbConn};

    #[tokio::test]
    async fn test_delete_expired_threads() {
        let db = DbConn::new_in_memory().await.unwrap();
        assert_eq!(db.delete_expired_ephemeral_threads().await.unwrap(), 0);

        let user_id = testutils::create_user(&db).await;
        let _ephemeral_thread_id = db.create_thread(user_id, true).await.unwrap();
        let non_ephemeral_thread_id = db.create_thread(user_id, false).await.unwrap();

        // Update the updated_at time to be 8 days ago for all threads
        sqlx::query!("UPDATE threads SET updated_at = DATETIME('now', '-8 days')",)
            .execute(&db.pool)
            .await
            .unwrap();

        // Only the ephemeral thread should be deleted
        assert_eq!(db.delete_expired_ephemeral_threads().await.unwrap(), 1);

        // The remaining thread should be the non-ephemeral thread
        let threads = db
            .list_threads(None, None, None, None, None, false)
            .await
            .unwrap();
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0].id, non_ephemeral_thread_id);

        // No threads are ephemeral
        let threads = db
            .list_threads(None, None, Some(true), None, None, false)
            .await
            .unwrap();
        assert_eq!(threads.len(), 0);
    }
}
