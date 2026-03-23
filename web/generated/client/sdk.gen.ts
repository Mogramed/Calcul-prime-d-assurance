import { request } from "./client.gen";
import type { PredictionInput, QuoteListResponse, QuoteResponse } from "./types.gen";

type RequestContext = {
  clientId: string;
  signal?: AbortSignal;
};

export async function createQuote(
  body: PredictionInput,
  context: RequestContext,
) {
  return request<QuoteResponse>("/quotes", {
    method: "POST",
    json: body,
    signal: context.signal,
    headers: {
      "X-Client-ID": context.clientId,
    },
  });
}

export async function listQuotes(context: RequestContext) {
  return request<QuoteListResponse>("/quotes", {
    method: "GET",
    signal: context.signal,
    headers: {
      "X-Client-ID": context.clientId,
    },
  });
}

export async function getQuote(
  quoteId: string,
  context: RequestContext,
) {
  return request<QuoteResponse>(`/quotes/${quoteId}`, {
    method: "GET",
    signal: context.signal,
    headers: {
      "X-Client-ID": context.clientId,
    },
  });
}
