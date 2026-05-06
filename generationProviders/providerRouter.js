// SlickCoherence music generation provider router.
// Frontend should keep calling /api/generate-music while this router decides what powers the generation.
// "mock" is active now. "external_placeholder" is reserved for future third-party AI music APIs.
// "slickcoherence_model_placeholder" is reserved for the future SlickCoherence-owned AI music model.

import {
  createGenerationJob as createMockGenerationJob,
  getGenerationStatus as getMockGenerationStatus,
  saveGeneratedTrack as saveMockGeneratedTrack,
  getGeneratedTracks as getMockGeneratedTracks
} from "./mockProvider.js";

const normalizeProviderKey = (provider) => String(provider || "").trim().toLowerCase();

export const getRequestedProviderKey = (requestedProvider) => {
  const fromRequest = normalizeProviderKey(requestedProvider);
  const fromEnv = normalizeProviderKey(process.env.MUSIC_GENERATION_PROVIDER);
  return fromRequest || fromEnv || "mock";
};

export const getProviderUnavailableResponse = (provider) => ({
  success: false,
  provider,
  message: "Selected music generation provider is not available yet."
});

export const createGenerationJob = async (payload = {}) => {
  const provider = getRequestedProviderKey(payload.provider);

  if (provider === "mock") {
    return createMockGenerationJob({ ...payload, provider });
  }

  if (provider === "external_placeholder" || provider === "slickcoherence_model_placeholder") {
    return getProviderUnavailableResponse(provider);
  }

  return getProviderUnavailableResponse(provider);
};

export const getGenerationStatus = async (jobId, provider = "mock") => {
  const providerKey = getRequestedProviderKey(provider);

  // For Phase 2, only mock jobs exist in memory. Real providers will route status checks here later.
  if (providerKey === "mock") {
    return getMockGenerationStatus(jobId);
  }

  if (providerKey === "external_placeholder" || providerKey === "slickcoherence_model_placeholder") {
    return getProviderUnavailableResponse(providerKey);
  }

  return getMockGenerationStatus(jobId);
};

export const saveGeneratedTrack = async (track = {}) => {
  const provider = getRequestedProviderKey(track.provider || "mock");

  // Keep library saving active through the mock provider until database persistence is introduced.
  if (provider === "mock" || provider === "external_placeholder" || provider === "slickcoherence_model_placeholder") {
    return saveMockGeneratedTrack(track);
  }

  return saveMockGeneratedTrack(track);
};

export const getGeneratedTracks = async (userId = "anonymous", provider = "mock") => {
  const providerKey = getRequestedProviderKey(provider);

  if (providerKey === "mock" || providerKey === "external_placeholder" || providerKey === "slickcoherence_model_placeholder") {
    return getMockGeneratedTracks(userId);
  }

  return getMockGeneratedTracks(userId);
};
