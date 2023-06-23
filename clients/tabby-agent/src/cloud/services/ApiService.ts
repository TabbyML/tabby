import type { CancelablePromise } from "../../generated/core/CancelablePromise";
import type { BaseHttpRequest } from "../../generated/core/BaseHttpRequest";

import type { DeviceTokenRequest } from "../models/DeviceTokenRequest";
import type { DeviceTokenResponse } from "../models/DeviceTokenResponse";
import type { DeviceTokenAcceptResponse } from "../models/DeviceTokenAcceptResponse";
import type { DeviceTokenRefreshResponse } from "../models/DeviceTokenRefreshResponse";

export class ApiService {
  constructor(public readonly httpRequest: BaseHttpRequest) {}

  /**
   * @returns DeviceTokenResponse Success
   * @throws ApiError
   */
  public deviceToken(body: DeviceTokenRequest): CancelablePromise<DeviceTokenResponse> {
    return this.httpRequest.request({
      method: "POST",
      url: "/device-token",
      body,
    });
  }

  /**
   * @param code
   * @returns DeviceTokenAcceptResponse Success
   * @throws ApiError
   */
  public deviceTokenAccept(query: { code: string }): CancelablePromise<DeviceTokenAcceptResponse> {
    return this.httpRequest.request({
      method: "POST",
      url: "/device-token/accept",
      query,
    });
  }

  /**
   * @param token
   * @returns DeviceTokenRefreshResponse Success
   * @throws ApiError
   */
  public deviceTokenRefresh(token: string): CancelablePromise<DeviceTokenRefreshResponse> {
    return this.httpRequest.request({
      method: "POST",
      url: "/device-token/refresh",
      headers: { Authorization: `Bearer ${token}` },
    });
  }

  /**
   * @param body object for anonymous usage tracking
   */
  public usage(body: any): CancelablePromise<any> {
    return this.httpRequest.request({
      method: "POST",
      url: "/usage",
      body,
    });
  }
}
