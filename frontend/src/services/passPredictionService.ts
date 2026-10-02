import type { PassPredictionRequestPayload, PassPredictionResponsePayload } from "../types";

const PREDICTION_URL = import.meta.env.VITE_PASS_PREDICTION_URL;

export class PassPredictionService {
  async isApiReady(): Promise<boolean> {
    try {
      const response = await fetch(new URL("../health", PREDICTION_URL), { signal: AbortSignal.timeout(10000) });
      const body = await response.json();
      return response.ok && body.model_loaded === true;
    } catch {
      return false;
    }
  }

  async predictPass(payload: PassPredictionRequestPayload): Promise<PassPredictionResponsePayload> {
    if (!PREDICTION_URL) {
      throw new Error("Missing VITE_PASS_PREDICTION_URL in frontend/.env.local");
    }

    const response = await fetch(PREDICTION_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || "Failed to get pass prediction");
    }

    return response.json();
  }
}
