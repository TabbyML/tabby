// FIXME: refactor env variables for running tests
export const isBrowser = !!process.env["IS_BROWSER"];
export const isTest = !!process.env["IS_TEST"];
export const testLogDebug = !!process.env["TEST_LOG_DEBUG"];
