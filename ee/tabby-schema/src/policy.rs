use juniper::ID;

use crate::{CoreError, Result};

#[derive(Debug, Clone)]
pub struct AccessPolicy {
    user_id: ID,
    is_admin: bool,
}

impl AccessPolicy {
    pub fn new(user_id: &ID, is_admin: bool) -> Self {
        Self {
            user_id: user_id.to_owned(),
            is_admin,
        }
    }

    pub fn check_delete_thread_messages(&self, user_id: &ID) -> Result<()> {
        if self.user_id != *user_id {
            return Err(CoreError::Forbidden(
                "You cannot delete messages in a thread that you do not own.",
            ));
        }

        Ok(())
    }

    pub fn check_update_thread_persistence(&self, user_id: &ID) -> Result<()> {
        if self.user_id != *user_id {
            return Err(CoreError::Forbidden(
                "You cannot update the persistence of a thread that you do not own.",
            ));
        }

        Ok(())
    }

    pub fn check_read_analytic(&self, users: &[ID]) -> Result<()> {
        const ERROR: Result<()> = Err(CoreError::Forbidden(
            "You must be admin to read other users' analytic data",
        ));

        if users.is_empty() && !self.is_admin {
            return ERROR;
        }

        if !self.is_admin {
            for id in users {
                if self.user_id != *id {
                    return ERROR;
                }
            }
        }

        Ok(())
    }

    pub fn check_read_source(&self, _source_id: &str) -> Result<()> {
        // FIXME(meng): implement this
        Ok(())
    }
}
