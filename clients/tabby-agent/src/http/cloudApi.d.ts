export interface paths {
  "/usage": {
    post: operations["usage"];
  };
}

export interface components {
  schemas: {
    UsageRequest: object;
  };
}

export interface operations {
  usage: {
    requestBody: {
      content: {
        "application/json": components["schemas"]["UsageRequest"];
      };
    };
    responses: {
      200: {
        content: never;
      };
    };
  };
}
