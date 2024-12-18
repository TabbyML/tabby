use async_trait::async_trait;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    notification::{Notification, NotificationRecipient, NotificationService},
    AsID, AsRowid, DbEnum, Result,
};

struct NotificationServiceImpl {
    db: DbConn,
}

pub fn create(db: DbConn) -> impl NotificationService {
    NotificationServiceImpl { db }
}

#[async_trait]
impl NotificationService for NotificationServiceImpl {
    async fn create(&self, recipient: NotificationRecipient, content: &str) -> Result<ID> {
        let id = self
            .db
            .create_notification(recipient.as_enum_str(), content)
            .await?;
        Ok(id.as_id())
    }

    async fn list(&self, user_id: &ID) -> Result<Vec<Notification>> {
        let notifications = self
            .db
            .list_notifications_within_7days(user_id.as_rowid()?)
            .await?;
        Ok(notifications.into_iter().map(|n| n.into()).collect())
    }

    async fn mark_read(&self, user_id: &ID, id: Option<&ID>) -> Result<()> {
        if let Some(id) = id {
            self.db
                .mark_notification_read(id.as_rowid()?, user_id.as_rowid()?)
                .await?;
        } else {
            self.db
                .mark_all_notifications_read_by_user(user_id.as_rowid()?)
                .await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use tabby_db::DbConn;
    use tabby_schema::{notification::NotificationService, AsID};

    use super::*;

    #[tokio::test]
    async fn test_admin_list() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user_id = db
            .create_user("test".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        let notification_id = service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();

        let notifications = service.list(&user_id).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].id, notification_id);
        assert_eq!(notifications[0].content, "admin_list");
        assert!(!notifications[0].read);
    }

    #[tokio::test]
    async fn test_admin_list_read() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user_id = db
            .create_user("test".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        let notification_id = service
            .create(NotificationRecipient::Admin, "admin_list_read")
            .await
            .unwrap();
        service
            .mark_read(&user_id, Some(&notification_id))
            .await
            .unwrap();

        let notifications = service.list(&user_id).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].id, notification_id);
        assert_eq!(notifications[0].content, "admin_list_read");
        assert!(notifications[0].read);
    }

    #[tokio::test]
    async fn test_admin_list_all() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user_id = db
            .create_user("test".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();
        service
            .create(NotificationRecipient::AllUser, "admin_list_all_user")
            .await
            .unwrap();

        let notifications = service.list(&user_id).await.unwrap();
        assert_eq!(notifications.len(), 2);
        assert_eq!(notifications[0].content, "admin_list");
        assert!(!notifications[0].read);
        assert_eq!(notifications[1].content, "admin_list_all_user");
        assert!(!notifications[1].read);
    }

    #[tokio::test]
    async fn test_admin_mark_all_read_admin() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user_id = db
            .create_user("test".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();

        service.mark_read(&user_id, None).await.unwrap();
        let notifications = service.list(&user_id).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(notifications[0].read);
    }

    #[tokio::test]
    async fn test_admin_mark_read_twice() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user_id = db
            .create_user("test".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        let notification_id = service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();

        service
            .mark_read(&user_id, Some(&notification_id))
            .await
            .unwrap();
        let notifications = service.list(&user_id).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(notifications[0].read);

        assert!(service
            .mark_read(&user_id, Some(&notification_id))
            .await
            .is_ok())
    }

    #[tokio::test]
    async fn test_admin_mark_all_read_twice() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user_id = db
            .create_user("test".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();

        service.mark_read(&user_id, None).await.unwrap();
        let notifications = service.list(&user_id).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(notifications[0].read);

        // mark all read will not return error even when call twice
        // but it should not create duplicated notifications
        service.mark_read(&user_id, None).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(notifications[0].read);
    }

    #[tokio::test]
    async fn test_admin_mark_all_read_admin_and_all_user() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user_id = db
            .create_user("test".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();
        service
            .create(NotificationRecipient::AllUser, "all_user")
            .await
            .unwrap();

        service.mark_read(&user_id, None).await.unwrap();
        let notifications = service.list(&user_id).await.unwrap();
        assert_eq!(notifications.len(), 2);
        assert!(notifications[0].read);
        assert!(notifications[1].read);
    }

    #[tokio::test]
    async fn test_user_mark_all_read_admin_and_all_user() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user_id = db
            .create_user("test".into(), None, false, None)
            .await
            .unwrap()
            .as_id();
        service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();
        service
            .create(NotificationRecipient::AllUser, "all_user")
            .await
            .unwrap();

        service.mark_read(&user_id, None).await.unwrap();
        let notifications = service.list(&user_id).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(notifications[0].read);
    }

    #[tokio::test]
    async fn test_multi_user_list() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user1 = db
            .create_user("test1".into(), None, false, None)
            .await
            .unwrap()
            .as_id();
        let user2 = db
            .create_user("test2".into(), None, false, None)
            .await
            .unwrap()
            .as_id();
        service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();
        service
            .create(NotificationRecipient::AllUser, "all_user")
            .await
            .unwrap();

        let notifications = service.list(&user1).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(!notifications[0].read);

        let notifications = service.list(&user2).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(!notifications[0].read);
    }

    #[tokio::test]
    async fn test_multi_user_mark_read() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user1 = db
            .create_user("test1".into(), None, false, None)
            .await
            .unwrap()
            .as_id();
        let user2 = db
            .create_user("test2".into(), None, false, None)
            .await
            .unwrap()
            .as_id();
        service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();
        service
            .create(NotificationRecipient::AllUser, "all_user")
            .await
            .unwrap();

        // user1 mark read
        service.mark_read(&user1, None).await.unwrap();
        let notifications = service.list(&user1).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(notifications[0].read);

        // user2 should still have unread notification
        let notifications = service.list(&user2).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(!notifications[0].read);

        // user2 mark read
        service.mark_read(&user2, None).await.unwrap();
        let notifications = service.list(&user2).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(notifications[0].read);
    }

    #[tokio::test]
    async fn test_multi_admin_mark_read() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user1 = db
            .create_user("test1".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        let user2 = db
            .create_user("test2".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();
        service
            .create(NotificationRecipient::AllUser, "all_user")
            .await
            .unwrap();

        // user1 mark read
        service.mark_read(&user1, None).await.unwrap();
        let notifications = service.list(&user1).await.unwrap();
        assert_eq!(notifications.len(), 2);
        assert!(notifications[0].read);
        assert!(notifications[1].read);

        // user2 should still have unread notification
        let notifications = service.list(&user2).await.unwrap();
        assert_eq!(notifications.len(), 2);
        assert!(!notifications[0].read);
        assert!(!notifications[1].read);

        // user2 mark read
        service.mark_read(&user2, None).await.unwrap();
        let notifications = service.list(&user2).await.unwrap();
        assert_eq!(notifications.len(), 2);
        assert!(notifications[0].read);
        assert!(notifications[1].read);
    }

    #[tokio::test]
    async fn test_multi_admin_user_mark_read() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let admin = db
            .create_user("test1".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        let user = db
            .create_user("test2".into(), None, false, None)
            .await
            .unwrap()
            .as_id();
        service
            .create(NotificationRecipient::Admin, "admin_list")
            .await
            .unwrap();
        service
            .create(NotificationRecipient::AllUser, "all_user")
            .await
            .unwrap();

        // admin mark read
        service.mark_read(&admin, None).await.unwrap();
        let notifications = service.list(&admin).await.unwrap();
        assert_eq!(notifications.len(), 2);
        assert!(notifications[0].read);
        assert!(notifications[1].read);

        // user should still have unread notification
        let notifications = service.list(&user).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(!notifications[0].read);

        // user mark read
        service.mark_read(&user, None).await.unwrap();
        let notifications = service.list(&user).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert!(notifications[0].read);
    }

    #[tokio::test]
    async fn test_admin_mark_single_then_all() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user = db
            .create_user("test1".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        let notification = db
            .create_notification("admin", "notification1")
            .await
            .unwrap()
            .as_id();
        db.create_notification("admin", "notification2")
            .await
            .unwrap();

        // mark single notification
        service.mark_read(&user, Some(&notification)).await.unwrap();
        let notifications = service.list(&user).await.unwrap();
        assert_eq!(notifications.len(), 2);
        assert!(notifications[0].read);
        assert!(!notifications[1].read);

        // mark all notifications
        service.mark_read(&user, None).await.unwrap();
        let notifications = service.list(&user).await.unwrap();
        assert_eq!(notifications.len(), 2);
        assert!(notifications[0].read);
        assert!(notifications[1].read);
    }

    #[tokio::test]
    async fn test_admin_mark_single_twice() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user = db
            .create_user("test1".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        let notification = db
            .create_notification("admin", "notification1")
            .await
            .unwrap()
            .as_id();

        service.mark_read(&user, Some(&notification)).await.unwrap();
        assert!(service.mark_read(&user, Some(&notification)).await.is_ok());
    }

    #[tokio::test]
    async fn test_admin_mark_all_twice() {
        let db = DbConn::new_in_memory().await.unwrap();
        let service = create(db.clone());

        let user = db
            .create_user("test1".into(), None, true, None)
            .await
            .unwrap()
            .as_id();
        db.create_notification("admin", "notification1")
            .await
            .unwrap()
            .as_id();

        service.mark_read(&user, None).await.unwrap();
        assert!(service.mark_read(&user, None).await.is_ok());
    }
}
