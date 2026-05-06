// SlickCoherence mock music generation provider.
// This safe provider is used for testing the full generation flow without a paid/external AI model.
// Future providers should expose the same function names so /api/generate-music stays stable.

const generationJobs = new Map();
const generatedTracks = [];

const buildGenerationJobId = () => `gen_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
const buildGeneratedTrackId = () => `track_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;

const getMockGenerationStatus = (pollCount = 0) => {
  if (pollCount >= 3) return "completed";
  if (pollCount >= 2) return "finalizing";
  if (pollCount >= 1) return "generating";
  return "queued";
};

const getBpmFromStylePrompt = (stylePrompt = "") => {
  const match = String(stylePrompt).match(/(\d{2,3})\s*bpm/i);
  const bpm = match ? Number(match[1]) : 120;
  return Number.isFinite(bpm) ? Math.max(60, Math.min(180, bpm)) : 120;
};

const getKeyFromStylePrompt = (stylePrompt = "") => {
  const prompt = String(stylePrompt).toLowerCase();
  const keys = [
    "A Minor", "B Minor", "C Minor", "D Minor", "E Minor", "F Minor", "G Minor",
    "A Major", "B Major", "C Major", "D Major", "E Major", "F Major", "G Major"
  ];
  const found = keys.find((key) => prompt.includes(key.toLowerCase()));
  return found || "A Minor";
};

export const mockProviderMetadata = {
  provider: "mock",
  providerLabel: "SlickCoherence Mock Provider",
  generationMode: "mock_preview",
  modelVersion: "mock-v1"
};

export const createGenerationJob = async (payload = {}) => {
  const {
    userId = "anonymous",
    title = "Untitled SlickCoherence Track",
    lyrics = "",
    stylePrompt = "",
    vocalGender = "mixed",
    lyricsMode = "auto",
    weirdness = 50,
    styleInfluence = 50,
    visibility = "Private"
  } = payload;

  const jobId = buildGenerationJobId();
  const job = {
    jobId,
    userId,
    title,
    lyrics,
    stylePrompt,
    vocalGender,
    lyricsMode,
    weirdness,
    styleInfluence,
    visibility,
    ...mockProviderMetadata,
    createdAt: new Date().toISOString(),
    status: "queued",
    pollCount: 0
  };

  generationJobs.set(jobId, job);

  return {
    success: true,
    ...mockProviderMetadata,
    jobId,
    status: job.status,
    message: "Music generation job created successfully.",
    estimatedTime: 30
  };
};

export const getGenerationStatus = async (jobId) => {
  const job = generationJobs.get(jobId);

  if (!job) {
    return { success: false, statusCode: 404, message: "Job not found" };
  }

  job.pollCount += 1;
  job.status = getMockGenerationStatus(job.pollCount);
  generationJobs.set(jobId, job);

  if (job.status === "completed") {
    const bpm = getBpmFromStylePrompt(job.stylePrompt);
    const key = getKeyFromStylePrompt(job.stylePrompt);

    return {
      success: true,
      jobId,
      status: "completed",
      audioUrl: "/api/mock-audio/slickcoherence-preview.wav",
      coverUrl: null,
      duration: "3:15",
      bpm,
      key,
      ...mockProviderMetadata,
      title: job.title,
      lyrics: job.lyrics,
      stylePrompt: job.stylePrompt,
      vocalGender: job.vocalGender,
      lyricsMode: job.lyricsMode,
      weirdness: job.weirdness,
      styleInfluence: job.styleInfluence,
      visibility: job.visibility,
      createdAt: job.createdAt
    };
  }

  return {
    success: true,
    jobId,
    status: job.status,
    ...mockProviderMetadata,
    message: `Generation status: ${job.status}`
  };
};

export const saveGeneratedTrack = async (trackData = {}) => {
  const track = {
    id: trackData.id || buildGeneratedTrackId(),
    ...trackData,
    provider: trackData.provider || mockProviderMetadata.provider,
    providerLabel: trackData.providerLabel || mockProviderMetadata.providerLabel,
    generationMode: trackData.generationMode || mockProviderMetadata.generationMode,
    modelVersion: trackData.modelVersion || mockProviderMetadata.modelVersion,
    savedAt: new Date().toISOString()
  };

  generatedTracks.unshift(track);

  return {
    success: true,
    message: "Generated track saved successfully.",
    track
  };
};

export const getGeneratedTracks = async (userId = "anonymous") => {
  const tracks = generatedTracks.filter((track) => !track.userId || track.userId === userId);
  return { success: true, tracks };
};
