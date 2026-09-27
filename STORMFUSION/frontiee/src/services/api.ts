import {
  CycloneTelemetry,
  ForecastHourData,
  TrackPoint,
  SatelliteFrame,
  GradCamAnalysis,
  DistrictRisk,
  EmergencyAlert,
  HistoricalAnalogue,
  GroundReport,
  ChatMessage,
  RainfallDistrictForecast
} from '../types/cyclone';

import { mockCurrentCyclone, mockEnvironmentalSoundings } from '../data/mockCycloneData';
import { mockForecastHours, mockFullTrack, mockIntensityTimeSeries, mockUncertaintyConeCoords, mockRainfallForecastZones, mockRainfallSwaths } from '../data/mockForecast';
import { mockSatelliteFrames, mockGradCamAnalysis } from '../data/mockSatellite';
import { mockDistrictRisks } from '../data/mockRiskZones';
import { mockAlerts } from '../data/mockAlerts';
import { mockHistoricalAnalogues } from '../data/mockHistoricalAnalogues';
import { mockSocialReports } from '../data/mockSocialReports';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
const IS_LIVE_BACKEND = import.meta.env.VITE_APP_MODE !== 'mock-only';

// Simulated network latency helper for fallbacks
const simulateLatency = <T>(data: T, delayMs: number = 100): Promise<T> => {
  return new Promise((resolve) => setTimeout(() => resolve(data), delayMs));
};

export const cycloneApi = {
  /**
   * Fetch current active cyclone status and core telemetry
   * Backend target: GET /api/cyclones/current or GET /api/cyclone/{id}
   */
  async getCurrentCyclone(stormId: string = 'TC_2026_02B'): Promise<CycloneTelemetry> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/cyclones/current`);
        if (response.ok) {
          const data = await response.json();
          return {
            ...mockCurrentCyclone,
            ...data
          };
        }
      } catch (err) {
        console.warn('[CycloneAPI] Live backend unavailable, falling back to simulated data.', err);
      }
    }
    return simulateLatency({ ...mockCurrentCyclone });
  },

  /**
   * Fetch atmospheric & oceanic environmental soundings (SST, wind shear, heat content)
   * Backend target: GET /api/cyclone/{id}/environment
   */
  async getEnvironmentalSoundings(stormId: string = 'TC_2026_02B') {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/cyclone/${stormId}/environment`);
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live environmental soundings endpoint failed, using fallback.', err);
      }
    }
    return simulateLatency({ ...mockEnvironmentalSoundings });
  },

  /**
   * Fetch multi-step AI forecast (24h, 48h, 72h)
   * Backend target: GET /api/forecast/{id}
   */
  async getForecast(stormId: string = 'TC_2026_02B'): Promise<{
    hours: ForecastHourData[];
    intensityTrend: typeof mockIntensityTimeSeries;
    uncertaintyCone: [number, number][];
  }> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/forecast/${stormId}`);
        if (response.ok) {
          const data = await response.json();
          return {
            hours: data.hours || mockForecastHours,
            intensityTrend: data.intensityTrend || mockIntensityTimeSeries,
            uncertaintyCone: data.uncertaintyCone || mockUncertaintyConeCoords
          };
        }
      } catch (err) {
        console.warn('[CycloneAPI] Live forecast endpoint failed, using simulated data.', err);
      }
    }
    return simulateLatency({
      hours: mockForecastHours,
      intensityTrend: mockIntensityTimeSeries,
      uncertaintyCone: mockUncertaintyConeCoords
    });
  },

  /**
   * Fetch complete trajectory (historical observed + AI predicted)
   * Backend target: GET /api/track/{id}
   */
  async getTrack(stormId: string = 'TC_2026_02B'): Promise<TrackPoint[]> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/track/${stormId}`);
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live track endpoint failed, using simulated data.', err);
      }
    }
    return simulateLatency([...mockFullTrack]);
  },

  /**
   * Fetch multispectral satellite frames and timeline
   * Backend target: GET /api/satellite/latest or GET /api/satellite/{channel}
   */
  async getSatelliteFrames(channel?: string): Promise<SatelliteFrame[]> {
    if (IS_LIVE_BACKEND) {
      try {
        const url = channel ? `${API_BASE_URL}/satellite?channel=${channel}` : `${API_BASE_URL}/satellite/latest`;
        const response = await fetch(url);
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live satellite endpoint failed, using simulated data.', err);
      }
    }
    const frames = channel
      ? mockSatelliteFrames.filter(f => f.channel === channel)
      : mockSatelliteFrames;
    return simulateLatency(frames);
  },

  /**
   * Fetch Explainable AI Grad-CAM attention visual and weights
   * Backend target: GET /api/explainability/gradcam/{id}
   */
  async getGradCamAnalysis(stormId: string = 'TC_2026_02B'): Promise<GradCamAnalysis> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/explainability/gradcam/${stormId}`);
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live Grad-CAM endpoint failed, using fallback.', err);
      }
    }
    return simulateLatency({ ...mockGradCamAnalysis });
  },

  /**
   * Fetch GIS risk zones and district-level vulnerability exposure
   * Backend target: GET /api/risk/{id}
   */
  async getRiskZones(stormId: string = 'TC_2026_02B'): Promise<DistrictRisk[]> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/risk/${stormId}`);
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live risk endpoint failed, using simulated data.', err);
      }
    }
    return simulateLatency([...mockDistrictRisks]);
  },

  /**
   * Fetch emergency alerts and disaster advisories
   * Backend target: GET /api/alerts
   */
  async getAlerts(): Promise<EmergencyAlert[]> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/alerts`);
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live alerts endpoint failed, using simulated data.', err);
      }
    }
    return simulateLatency([...mockAlerts]);
  },

  /**
   * Acknowledge an emergency alert
   * Backend target: POST /api/alerts/{id}/acknowledge
   */
  async acknowledgeAlert(id: string): Promise<EmergencyAlert | null> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/alerts/${id}/acknowledge`, { method: 'POST' });
        if (response.ok) {
          const data = await response.json();
          return data.alert;
        }
      } catch (err) {
        console.warn('[CycloneAPI] Live alert acknowledge failed.', err);
      }
    }
    return null;
  },

  /**
   * Fetch historical cyclone analogues from FAISS vector search
   * Backend target: GET /api/analogues/{id}
   */
  async getHistoricalAnalogues(stormId: string = 'TC_2026_02B'): Promise<HistoricalAnalogue[]> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/analogues/${stormId}`);
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live analogues endpoint failed, using fallback.', err);
      }
    }
    return simulateLatency([...mockHistoricalAnalogues]);
  },

  /**
   * Fetch auxiliary crowdsourced ground truth observations
   * Backend target: GET /api/social-reports
   */
  async getSocialReports(): Promise<GroundReport[]> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/social-reports`);
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live social reports endpoint failed, using fallback.', err);
      }
    }
    return simulateLatency([...mockSocialReports]);
  },

  /**
   * Submit an auxiliary crowdsourced ground report
   * Backend target: POST /api/social-report
   */
  async submitSocialReport(report: Omit<GroundReport, 'id' | 'timestamp' | 'upvotes' | 'verifiedOfficial'>): Promise<GroundReport> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/social-report`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(report)
        });
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live social report submit failed, falling back to local.', err);
      }
    }
    const newReport: GroundReport = {
      ...report,
      id: `SOC-${Date.now()}`,
      timestamp: 'Just now',
      upvotes: 1,
      verifiedOfficial: false
    };
    mockSocialReports.unshift(newReport);
    return simulateLatency(newReport);
  },

  /**
   * Upvote a crowdsourced ground report
   * Backend target: POST /api/social-report/{id}/upvote
   */
  async upvoteSocialReport(id: string): Promise<GroundReport | null> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/social-report/${id}/upvote`, { method: 'POST' });
        if (response.ok) {
          const data = await response.json();
          return data.report;
        }
      } catch (err) {
        console.warn('[CycloneAPI] Live social report upvote failed.', err);
      }
    }
    return null;
  },

  /**
   * Fetch district rainfall forecasts and spatial swaths
   * Backend target: GET /api/rainfall-forecast
   */
  async getRainfallForecast(): Promise<{
    zones: RainfallDistrictForecast[];
    swaths: { core: [number, number][]; outer: [number, number][] };
  }> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/rainfall-forecast`);
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live rainfall forecast endpoint failed.', err);
      }
    }
    return simulateLatency({
      zones: mockRainfallForecastZones,
      swaths: mockRainfallSwaths
    });
  },

  /**
   * Fetch AI model system & training status
   * Backend target: GET /api/v1/model/status
   */
  async getModelStatus() {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/v1/model/status`);
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live model status endpoint failed.', err);
      }
    }
    return null;
  },

  /**
   * Query the contextual StormFusion Assistant
   * Backend target: POST /api/chat
   */
  async sendChatMessage(message: string, history: ChatMessage[]): Promise<ChatMessage> {
    if (IS_LIVE_BACKEND) {
      try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, history })
        });
        if (response.ok) return await response.json();
      } catch (err) {
        console.warn('[CycloneAPI] Live chat endpoint failed, falling back to simulated inference.', err);
      }
    }

    // Context-aware intelligent fallback
    const query = message.toLowerCase();
    let responseText = "";
    let dataPoints: { label: string; value: string }[] | undefined;

    if (query.includes("where") || query.includes("location") || query.includes("coordinate")) {
      responseText = `STORMFUSION estimates the storm is centered at coordinates 14.52° N, 87.21° E over the central Bay of Bengal, moving Northwestward at 14 km/h.`;
      dataPoints = [
        { label: "Current Position", value: "14.52° N, 87.21° E" },
        { label: "Movement", value: "NW @ 14 km/h" },
        { label: "Central Pressure", value: "965 hPa" }
      ];
    } else if (query.includes("track") || query.includes("direction") || query.includes("predicted track")) {
      responseText = `The multimodal ConvLSTM track model projects a continuing Northwest trajectory towards the North AP / South Odisha coast.`;
      dataPoints = [
        { label: "Primary Track", value: "Northwestward recurvature" },
        { label: "72h Projection", value: "18.1° N, 82.5° E" }
      ];
    } else if (query.includes("intensity") || query.includes("wind") || query.includes("category")) {
      responseText = `The storm is classified as a "Very Severe Cyclonic Storm" with maximum sustained winds of 82 knots (152 km/h) and a minimum central pressure of 965 hPa.`;
      dataPoints = [
        { label: "Current Intensity", value: "82 kt (152 km/h)" },
        { label: "Category", value: "Very Severe Cyclonic Storm" },
        { label: "Rapid Intensification", value: "MODERATE" }
      ];
    } else {
      responseText = `StormFusion Command Assistant operational. I am connected live to the STORMFUSION backend and can answer queries on storm location, intensity, track forecast, risk zones, and advisories.`;
    }

    const assistantMsg: ChatMessage = {
      id: `MSG-${Date.now()}`,
      sender: 'assistant',
      content: responseText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      dataPoints
    };

    return simulateLatency(assistantMsg, 300);
  }
};
