import type { BaseHttpRequest, OpenAPIConfig } from "../generated";
import { AxiosHttpRequest } from "../generated/core/AxiosHttpRequest";
import { ApiService } from "./services/ApiService";

type HttpRequestConstructor = new (config: OpenAPIConfig) => BaseHttpRequest;

export class CloudApi {
  public readonly api: ApiService;

  public readonly request: BaseHttpRequest;

  constructor(config?: Partial<OpenAPIConfig>, HttpRequest: HttpRequestConstructor = AxiosHttpRequest) {
    this.request = new HttpRequest({
      BASE: config?.BASE,
      VERSION: config?.VERSION ?? "0.0.0",
      WITH_CREDENTIALS: config?.WITH_CREDENTIALS ?? false,
      CREDENTIALS: config?.CREDENTIALS ?? "include",
      TOKEN: config?.TOKEN,
      USERNAME: config?.USERNAME,
      PASSWORD: config?.PASSWORD,
      HEADERS: config?.HEADERS,
      ENCODE_PATH: config?.ENCODE_PATH,
    });

    this.api = new ApiService(this.request);
  }
}
