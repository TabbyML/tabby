"use client"

import useSWRImmutable from 'swr/immutable';
import { SWRResponse } from 'swr'

export interface HealthInfo {
    device: string,
    model: string,
    chat_model?: string,
    version: {
        build_date: string,
        git_describe: string,
    }
}

export function useHealth(): SWRResponse<HealthInfo> {
    return useSWRImmutable('/v1/health', {
        fallbackData: process.env.NODE_ENV !== "production" ? {
            "device": "metal",
            "model": "TabbyML/StarCoder-1B",
            "version": {
                "build_date": "2023-10-21",
                "git_describe": "v0.3.1",
                "git_sha": "d5fdcf3a2cbe0f6b45d6e8ef3255e6a18f840132"
            }
        } : undefined
    });
}