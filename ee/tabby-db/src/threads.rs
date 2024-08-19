use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{query, query_as, types::Json, FromRow};
use tabby_db_macros::query_paged_as;

use crate::DbConn;

#[derive(FromRow)]
pub struct ThreadDAO {
    pub id: i64,
    pub user_id: i64,
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

    pub code_attachments: Option<Json<Vec<ThreadMessageAttachmentCode>>>,
    pub doc_attachments: Option<Json<Vec<ThreadMessageAttachmentDoc>>>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize)]
pub struct ThreadMessageAttachmentDoc {
    pub title: String,
    pub link: String,
    pub content: String,
}

#[derive(Serialize, Deserialize)]
pub struct ThreadMessageAttachmentCode {
    pub git_url: String,
    pub language: String,
    pub filepath: String,
    pub content: String,
    pub start_line: usize,
}

impl DbConn {
    pub async fn create_thread(&self, user_id: i64) -> Result<i64> {
        let res = query!("INSERT INTO threads(user_id) VALUES (?)", user_id)
            .execute(&self.pool)
            .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn list_threads(
        &self,
        ids: Option<&[i64]>,
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

        let condition = (!conditions.is_empty()).then_some(conditions.join(" AND "));
        let threads = query_paged_as!(
            ThreadDAO,
            "threads",
            [
                "id",
                "user_id",
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
            "UPDATE threads SET relevant_questions = ? WHERE id = ?",
            relevant_questions,
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
        code_attachments: Option<&[ThreadMessageAttachmentCode]>,
        doc_attachments: Option<&[ThreadMessageAttachmentDoc]>,
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
        let doc_attachments = doc_attachments.map(Json);
        let res = query!(
            "INSERT INTO thread_messages(thread_id, role, content, code_attachments, doc_attachments) VALUES (?, ?, ?, ?, ?)",
            thread_id,
            role,
            content,
            code_attachments,
            doc_attachments,
        )
        .execute(&self.pool)
        .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn update_thread_message_attachments(
        &self,
        message_id: i64,
        code_attachments: Option<&[ThreadMessageAttachmentCode]>,
        doc_attachments: Option<&[ThreadMessageAttachmentDoc]>,
    ) -> Result<()> {
        let code_attachments = code_attachments.map(Json);
        let doc_attachments = doc_attachments.map(Json);
        query!(
            "UPDATE thread_messages SET code_attachments = ?, doc_attachments = ? WHERE id = ?",
            code_attachments,
            doc_attachments,
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
            "UPDATE thread_messages SET content = content || ? WHERE id = ?",
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
                code_attachments as "code_attachments: Json<Vec<ThreadMessageAttachmentCode>>",
                doc_attachments as "doc_attachments: Json<Vec<ThreadMessageAttachmentDoc>>",
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
                "code_attachments" as "code_attachments: Json<Vec<ThreadMessageAttachmentCode>>",
                "doc_attachments" as "doc_attachments: Json<Vec<ThreadMessageAttachmentDoc>>",
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
}
