// // Inspired by
// // https://github.com/urql-graphql/urql/tree/main/exchanges/auth

// import type { Source } from 'wonka';
// import {
//   pipe,
//   map,
//   filter,
//   onStart,
//   take,
//   makeSubject,
//   toPromise,
//   merge,
//   fromPromise,
//   mergeMap
// } from 'wonka';

// import type {
//   Operation,
//   OperationContext,
//   OperationResult,
//   CombinedError,
//   Exchange,
//   DocumentInput,
//   AnyVariables,
//   OperationInstance,
// } from '@urql/core';
// import { createRequest, makeOperation, makeErrorResult } from '@urql/core';
// import { Fetcher, Key as SWRKey, SWRResponse } from 'swr'

// type TFetchConfig = {
//   // todo just string so far
//   fetchKey: string
//   init?: RequestInit
//   context?: { authAttempt?: boolean }
// }

// type TFetchResult = {
//   res: Response,
//   config: TFetchConfig
//   error?: any
// }

// function _markErrorResult(config: TFetchConfig, error: Error, res?: Response) {
//   return {
//     config,
//     res
//   }
// }

// /** Utilities to use while refreshing authentication tokens. */
// interface AuthUtilities {
//   mutate<Data = any>(
//     key: string
//   ): Promise<SWRResponse<Data>>;
//   appendHeaders(
//     fetchConfig: TFetchConfig,
//     headers: Record<string, string>
//   ): TFetchConfig;
// }

// /** Configuration for the `authExchange` returned by the initializer function you write. */
// interface AuthConfig {
//   addAuthToFetch(fetchConfig: TFetchConfig): TFetchConfig;
//   willAuthError?(fetchConfig: TFetchConfig): boolean;
//   didAuthError(error: any, fetchConfig: TFetchConfig): boolean;
//   refreshAuth(): Promise<void>;
// }

// const addAuthAttemptToRestfulApi = (
//   config: TFetchConfig,
//   authAttempt: boolean
// ): TFetchConfig => {
//   // makeOperation(operation.kind, operation, {
//   //   ...operation.context,
//   //   authAttempt,
//   // });
//   return {
//     ...config,
//     context: {
//       ...(config.context || {}),
//       authAttempt
//     }
//   }
// }

// const bypassQueue = new Set<TFetchConfig | undefined>();
// const retries = makeSubject<TFetchConfig>();
// const errors = makeSubject<any>();

// let retryQueue = new Map<string, TFetchConfig>();
// let authPromise: Promise<void> | void;

// export function authExchange(
//   init: (utilities: AuthUtilities) => Promise<AuthConfig>
// ) {
//   // return () => {

//     function flushQueue() {
//       authPromise = undefined;
//       const queue = retryQueue;
//       retryQueue = new Map();
//       queue.forEach(retries.next);
//     }

//     function errorQueue(error: Error) {
//       authPromise = undefined;
//       const queue = retryQueue;
//       retryQueue = new Map();
//       queue.forEach(fetchConfig => {
//         errors.next(_markErrorResult(fetchConfig, error));
//       });
//     }

//     let config: AuthConfig | null = null;

//     return () => {
//       function initAuth() {
//         authPromise = Promise.resolve()
//           .then(() =>
//             init({
//               mutate<Data>(k: string): Promise<any> {
//                 // const baseOperation = client.createRequestOperation(
//                 //   'mutation',
//                 //   createRequest(query, variables),
//                 //   context
//                 // );
//                 const baseRequest: TFetchConfig = { fetchKey: k }
//                 return pipe(
//                   result$,
//                   onStart(() => {
//                     const fetchConfig = addAuthToFetch(baseRequest);
//                     // bypassQueue.add(
//                     //   operation.context._instance as OperationInstance
//                     // );
//                     bypassQueue.add(fetchConfig)
//                     retries.next(fetchConfig);
//                   }),
//                   // filter(
//                   //   result =>
//                   //     result.operation.key === baseRequest.key &&
//                   //     baseRequest.context._instance ===
//                   //       result.operation.context._instance
//                   // ),
//                   filter(result => {
//                     return result.
//                   }),
//                   take(1),
//                   toPromise
//                 );
//               },
//               appendHeaders(
//                 fetchConfig: TFetchConfig,
//                 headers: Record<string, string>
//               ) {
//                 return {
//                   ...fetchConfig,
//                   init: {
//                     ...fetchConfig.init,
//                     // todo more headers
//                     headers
//                   }
//                 }
//               },
//             })
//           )
//           .then((_config: AuthConfig) => {
//             if (_config) config = _config;
//             flushQueue();
//           })
//           .catch((error: Error) => {
//             if (process.env.NODE_ENV !== 'production') {
//               console.warn(
//                 'authExchange()’s initialization function has failed, which is unexpected.\n' +
//                   'If your initialization function is expected to throw/reject, catch this error and handle it explicitly.\n' +
//                   'Unless this error is handled it’ll be passed onto any `OperationResult` instantly and authExchange() will block further operations and retry.',
//                 error
//               );
//             }

//             errorQueue(error);
//           });
//       }
//       console.log('====call init')
//       initAuth();

//       function refreshAuth(fetchConfig: TFetchConfig) {
//         // add to retry queue to try again later
//         retryQueue.set(
//           fetchConfig.fetchKey,
//           addAuthAttemptToRestfulApi(fetchConfig, true)
//         );

//         // check that another operation isn't already doing refresh
//         if (config && !authPromise) {
//           authPromise = config.refreshAuth().then(flushQueue).catch(errorQueue);
//         }
//       }

//       function willAuthError(fetchConfig: TFetchConfig) {
//         return (
//           !fetchConfig?.context?.authAttempt &&
//           config &&
//           config.willAuthError &&
//           config.willAuthError(fetchConfig)
//         );
//       }

//       function didAuthError(result: TFetchResult) {
//         return (
//           config &&
//           config.didAuthError &&
//           config.didAuthError(result.error, result.config)
//         );
//       }

//       function addAuthToFetch(fetchConfig: TFetchConfig) {
//         return config ? config.addAuthToFetch(fetchConfig) : fetchConfig;
//       }

//       const opsWithAuth$ = pipe(
//         // merge([retries.source, operations$]),
//         retries.source,
//         map(fetchConfig => {
//           if (
//             fetchConfig.fetchKey &&
//             bypassQueue.has(fetchConfig)
//           ) {
//             return fetchConfig;
//           } else if (fetchConfig?.context?.authAttempt) {
//             return addAuthToFetch(fetchConfig);
//           } else if (authPromise || !config) {
//             if (!authPromise) initAuth();

//             if (!retryQueue.has(fetchConfig.fetchKey))
//               retryQueue.set(
//                 fetchConfig.fetchKey,
//                 addAuthAttemptToRestfulApi(fetchConfig, false)
//               );

//             return null;
//           } else if (willAuthError(fetchConfig)) {
//             refreshAuth(fetchConfig);
//             return null;
//           }

//           return addAuthToFetch(
//             addAuthAttemptToRestfulApi(fetchConfig, false)
//           );
//         }),
//         filter(Boolean)
//       ) as Source<TFetchConfig>;

//       // todo forward 可以是传入的 onComplete
//       // const result$ = pipe(opsWithAuth$, forward);
//       const result$ = opsWithAuth$;

//       return merge([
//         errors.source,
//         pipe(
//           result$,
//           map(fetchConfig => fromPromise<Response>(fetch(fetchConfig.fetchKey, fetchConfig.init).then(res => res.json()))),
//           map(response => {
//             return {
//               res: response,
//               error: response.error,
//               config: {} as TFetchConfig
//             } as TFetchResult
//           }),
//           filter(result => {
//             if (
//               !bypassQueue.has(result) &&
//               didAuthError(result) &&
//               !result.operation.context.authAttempt
//             ) {
//               refreshAuth(result);
//               return false;
//             }

//             if (bypassQueue.has(result.operation.context._instance)) {
//               bypassQueue.delete(result.operation.context._instance);
//             }

//             return true;
//           })
//         ),
//       ]);
//     };
//   // };
// }
