import type { CancelablePromise } from "../../generated/core/CancelablePromise";
import type { BaseHttpRequest } from "../../generated/core/BaseHttpRequest";

import { DeviceTokenResponse } from "../models/DeviceTokenResponse";
import { DeviceTokenAcceptResponse } from "../models/DeviceTokenAcceptResponse";

export class ApiService {
  constructor(public readonly httpRequest: BaseHttpRequest) {}

  /**
   * @returns DeviceTokenResponse Success
   * @throws ApiError
   */
  public deviceToken(): CancelablePromise<DeviceTokenResponse> {
    return this.httpRequest.request({
      method: "POST",
      url: "/api/device-token",
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
      url: "/api/device-token/accept",
      query,
    });
  }
}
