import React, { useState, useEffect, useRef } from "react";
import { 
  Shield, 
  Upload, 
  FileText, 
  AlertTriangle, 
  CheckCircle, 
  Lock, 
  LogOut, 
  History, 
  Search,
  ChevronRight,
  Eye,
  EyeOff,
  RefreshCw,
  Activity,
  Download,
  User as UserIcon,
  Key,
  Trash2,
  Archive,
  Printer,
  Loader2
} from "lucide-react";
import { Toaster, toast } from "sonner";
import { analyzeDocument, AnalysisResult, RLWeights } from "./services/geminiService";
import { User, ScanRecord } from "./types";
import { cn } from "../lib/utils";

// --- Custom Internal Components for High Stability ---

interface AnalysisStep {
  id: string;
  label: string;
  status: "pending" | "loading" | "completed" | "error";
  icon: React.ReactNode;
  subtasks: string[];
  description: string;
}

interface ProcessLogEntry {
  id: string;
  message: string;
  type: "info" | "success" | "warning" | "error" | "running" | "executing";
  timestamp: string;
  processName?: string;
}

const HeatmapOverlay = ({ regions, show }: { regions: { x: number; y: number; width: number; height: number; reason: string }[], show: boolean }) => {
  if (!show || !regions) return null;
  return (
    <div className="absolute inset-0 pointer-events-none z-20 overflow-hidden">
      {regions.map((region, i) => (
        <div
          key={i}
          className="absolute border-2 border-rose-500 bg-rose-500/10 rounded shadow-[0_0_12px_rgba(244,63,94,0.5)] animate-pulse"
          style={{
            left: `${region.x}%`,
            top: `${region.y}%`,
            width: `${region.width}%`,
            height: `${region.height}%`,
          }}
        >
          <div className="absolute -top-7 left-0 bg-rose-600 text-white text-[9px] px-2 py-0.5 rounded font-mono font-bold whitespace-nowrap shadow-md">
            {region.reason}
          </div>
        </div>
      ))}
    </div>
  );
};

const ForensicScoreGauge = ({ score }: { score: number }) => {
  const radius = 45;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const color = score > 70 ? "#10b981" : score > 40 ? "#06b6d4" : "#f43f5e";
  
  return (
    <div className="relative w-48 h-48 mx-auto flex items-center justify-center">
      {/* Visual cyber decorations around gauge */}
      <div className="absolute inset-0 rounded-full border border-slate-800/60 scale-[1.1]" />
      <div className="absolute inset-0 rounded-full border border-dashed border-cyan-500/10 scale-[1.25] animate-spin" style={{ animationDuration: '40s' }} />
      
      <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="#1e293b"
          strokeWidth="8"
        />
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
          style={{ filter: `drop-shadow(0 0 6px ${color}80)` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-extrabold tracking-tight font-mono text-white select-none">{score}%</span>
        <span className="text-[9px] font-semibold text-slate-500 uppercase tracking-widest mt-1">Trust Score</span>
      </div>
    </div>
  );
};

const ForensicBarChart = ({ score }: { score: number }) => {
  const dataset = [
    { label: "METADATA", score: Math.round(score > 70 ? 92 + (100 - score) / 4 : score * 0.95) },
    { label: "FORENSICS", score: Math.round(score > 50 ? score * 1.02 : score * 0.82) },
    { label: "NEURAL NETWORK", score: score },
    { label: "BIOMETRIC", score: Math.round(score > 60 ? 82 + (100 - score) / 5 : score * 0.88) }
  ];

  return (
    <div className="w-full space-y-4">
      {dataset.map((data, idx) => {
        const barColor = data.score > 70 ? "bg-emerald-500/80" : data.score > 40 ? "bg-cyan-500/80" : "bg-rose-500/80";
        return (
          <div key={idx} className="space-y-1.5">
            <div className="flex justify-between items-center text-[10px] font-mono">
              <span className="text-slate-400 font-bold tracking-widest">{data.label}</span>
              <span className="text-white font-extrabold">{data.score}%</span>
            </div>
            <div className="h-2 w-full bg-slate-905 rounded-full overflow-hidden border border-slate-800 relative">
              <div 
                className={`h-full ${barColor} rounded-full transition-all duration-1000 ease-out`}
                style={{ width: `${data.score}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [view, setView] = useState<"login" | "dashboard" | "analysis">("login");
  const [loading, setLoading] = useState(false);
  
  // Reinforcement Learning parameters
  const [rlWeights, setRlWeights] = useState<RLWeights>({
    realDocBias: 5.0,              // dynamically balances underestimations
    strictnessModifier: 0.85,     // fine-tunes harsh deductions
    overallCalibration: 2.0,       // score baseline shift
    feedbackCount: 0,              // feedback logs count
    learningRate: 0.15             // step scaling
  });

  interface RLEpisode {
    episode: number;
    reward: number;
    feedbackType: string;
    previousWeights: RLWeights;
    newWeights: RLWeights;
    timestamp: string;
  }
  const [rlHistory, setRlHistory] = useState<RLEpisode[]>([]);

  const [currentFile, setCurrentFile] = useState<{ path: string; name: string; mimeType: string; base64: string } | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [emailError, setEmailError] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [activeTab, setActiveTab] = useState<"findings" | "suggestions">("findings");

  // Multi-step scanning feedback simulation states
  const [activeSubtask, setActiveSubtask] = useState("");
  const [processLogs, setProcessLogs] = useState<ProcessLogEntry[]>([]);
  const [overallProgress, setOverallProgress] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState(16);

  const [analysisSteps, setAnalysisSteps] = useState<AnalysisStep[]>([
    { 
      id: "metadata", 
      label: "METADATA", 
      status: "pending", 
      icon: <Activity className="w-5 h-5" />,
      description: "Extracting file metadata, analyzing cryptographic signatures, verifying structural consistency.",
      subtasks: ["Extracting EXIF headers", "Validating system markers", "Checking timestamp coherence"]
    },
    { 
      id: "forensics", 
      label: "FORENSICS", 
      status: "pending", 
      icon: <FileText className="w-5 h-5" />,
      description: "Checking pixel ELA pattern deviations, JPEG double compression bounds, and GAN anomalies.",
      subtasks: ["Ela mapping scan", "Double jpeg artifact extraction", "Neural noise delta analysis"]
    },
    { 
      id: "neural", 
      label: "NEURAL NETWORK", 
      status: "pending", 
      icon: <Shield className="w-5 h-5" />,
      description: "Interrogating document layout, typography character bleed, and semantic flow vectors via Gemini.",
      subtasks: ["Kerning validation", "Document layout alignment matrix", "Synthetic layout model query"]
    },
    { 
      id: "biometric", 
      label: "BIOMETRICS", 
      status: "pending", 
      icon: <Lock className="w-5 h-5" />,
      description: "Analyzing skin textural pattern continuity and facial reflection vectors for deepfake markers.",
      subtasks: ["Dynamic pose scan", "Pixel boundaries inspection", "Synthetic vector alignment test"]
    }
  ]);

  // Load state and cache data
  useEffect(() => {
    try {
      const savedUser = localStorage.getItem("deepguard_user");
      if (savedUser) {
        setUser(JSON.parse(savedUser));
        setView("dashboard");
      }
    } catch (e) {
      console.error("Corrupted user session in storage", e);
      localStorage.removeItem("deepguard_user");
    }

    try {
      const savedScans = localStorage.getItem("deepguard_scans");
      if (savedScans) {
        setScans(JSON.parse(savedScans));
      } else {
        localStorage.setItem("deepguard_scans", JSON.stringify([]));
      }
    } catch (e) {
      console.error("Corrupted scans dataset", e);
      setScans([]);
    }

    try {
      const savedWeights = localStorage.getItem("deepguard_rl_weights");
      if (savedWeights) {
        setRlWeights(JSON.parse(savedWeights));
      }
      const savedHistory = localStorage.getItem("deepguard_rl_history");
      if (savedHistory) {
        setRlHistory(JSON.parse(savedHistory));
      }
    } catch (e) {
      console.error("Corrupted RL dataset in cache", e);
    }
  }, []);

  const validateEmail = (candidate: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(candidate);
  };

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateEmail(email)) {
      setEmailError("INVALID_AGENT_ID: Forensic agent email format invalid.");
      return;
    }

    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasNonalphas = /\W/.test(password);
    const isLongEnough = password.length >= 8;

    if (!(hasUpperCase && hasLowerCase && hasNumbers && hasNonalphas && isLongEnough)) {
      setPasswordError("WEAK_ENCRYPTION_KEY: Must contain 8+ characters, including upper, lower, number, and symbol.");
      return;
    }

    setEmailError("");
    setPasswordError("");
    const name = email.split("@")[0];
    const authenticatedUser: User = { id: Date.now().toString(), email, name };
    setUser(authenticatedUser);
    localStorage.setItem("deepguard_user", JSON.stringify(authenticatedUser));
    setView("dashboard");
    toast.success(`Access authorized. Welcome back, Agent ${name}.`);
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem("deepguard_user");
    setView("login");
    toast.info("Session terminated. Connection closed.");
  };

  const deleteScan = (id: string) => {
    const updated = scans.filter(s => s.id !== id);
    setScans(updated);
    localStorage.setItem("deepguard_scans", JSON.stringify(updated));
    toast.success("Forensic archive record deleted.");
  };

  const clearArchive = () => {
    setScans([]);
    localStorage.setItem("deepguard_scans", JSON.stringify([]));
    toast.success("Evidence log fully purged.");
  };

  const runAnalysisPipeline = async (base64: string, mimeType: string, fileName: string, filePath: string) => {
    setLoading(true);
    setProcessLogs([]);
    setOverallProgress(0);
    setEstimatedTime(16);

    const logProcess = (message: string, type: ProcessLogEntry["type"], processName: string) => {
      setProcessLogs(prev => [...prev, {
        id: Math.random().toString(36).substring(2),
        message,
        type,
        timestamp: new Date().toLocaleTimeString(),
        processName
      }]);
    };

    // Reset verification steps to pending
    setAnalysisSteps(prev => prev.map(s => ({ ...s, status: "pending" })));

    try {
      const stepDuration = 2000; // Duration per pipeline node
      
      for (let i = 0; i < analysisSteps.length; i++) {
        const step = analysisSteps[i];
        
        // Start Step Loading State
        setAnalysisSteps(prev => prev.map(s => s.id === step.id ? { ...s, status: "loading" } : s));
        logProcess(`Initializing forensic verification for node ${step.label}...`, "executing", step.label);

        const subtaskDuration = stepDuration / step.subtasks.length;
        for (let j = 0; j < step.subtasks.length; j++) {
          const subtask = step.subtasks[j];
          setActiveSubtask(subtask);
          logProcess(`Analyzing vectors: ${subtask}`, "running", step.label);
          
          await new Promise(resolve => setTimeout(resolve, subtaskDuration));
          
          setOverallProgress(prev => Math.min(prev + Math.round(100 / (analysisSteps.length * step.subtasks.length)), 95));
          setEstimatedTime(prev => Math.max(0, prev - (subtaskDuration / 1000)));
        }

        // Complete Step Loading State
        setAnalysisSteps(prev => prev.map(s => s.id === step.id ? { ...s, status: "completed" } : s));
        logProcess(`${step.label} verification node completed successfully.`, "success", step.label);
      }

      // Final step: calling the actual forensic Gemini API directly
      logProcess("Transmitting payload package to Central Forensic Intelligence...", "executing", "UPLINK");
      const result = await analyzeDocument(base64, mimeType, rlWeights);
      
      setAnalysisResult(result);
      logProcess("Gemini analysis payload successfully processed. Report generated.", "success", "REPORTS");

      const newRecord: ScanRecord = {
        id: Date.now().toString(),
        fileName,
        filePath,
        timestamp: new Date().toISOString(),
        trustScore: result.trust_score,
        classification: result.classification
      };

      const updatedScans = [newRecord, ...scans];
      setScans(updatedScans);
      localStorage.setItem("deepguard_scans", JSON.stringify(updatedScans));

      setOverallProgress(100);
      setEstimatedTime(0);
      
      await new Promise(resolve => setTimeout(resolve, 800));
      setView("analysis");
      setLoading(false);
    } catch (e: any) {
      console.error(e);
      toast.error("Forensic node integrity failed: " + (e.message || "Unknown anomaly"));
      setLoading(false);
    }
  };

  const applyRLFeedback = (feedbackType: string, rewardValue: number) => {
    const prevWeights = { ...rlWeights };
    
    // Learning adjustments based on policy gradients
    let deltaRealBias = 0;
    let deltaStrictness = 0;
    let deltaCalibration = 0;
    
    // Learning Rate
    const lr = prevWeights.learningRate;
    
    if (feedbackType === "underestimated" || feedbackType === "too_strict") {
      // Trust score was too low for a genuine document
      deltaRealBias = lr * 18; 
      deltaStrictness = -lr * 0.35; 
      deltaCalibration = lr * 6; 
    } else if (feedbackType === "excellent" || feedbackType === "accurate") {
      // Model worked perfectly, reinforce current parameters
      deltaRealBias = lr * 1.5 * (rewardValue / 10);
      deltaCalibration = lr * 0.8 * (rewardValue / 10);
    } else if (feedbackType === "false_positive" || feedbackType === "too_lenient") {
      // Model failed to catch synthetic edit / trust score too high
      deltaRealBias = -lr * 14;
      deltaStrictness = lr * 0.45;
      deltaCalibration = -lr * 8;
    } else if (feedbackType === "harsh") {
      // Verdict correct, but score too aggressive
      deltaRealBias = lr * 6;
      deltaStrictness = -lr * 0.15;
    }

    const updatedWeights: RLWeights = {
      realDocBias: Math.round((prevWeights.realDocBias + deltaRealBias) * 100) / 100,
      strictnessModifier: Math.round(Math.max(0.3, Math.min(2.5, prevWeights.strictnessModifier + deltaStrictness)) * 100) / 100,
      overallCalibration: Math.round((prevWeights.overallCalibration + deltaCalibration) * 100) / 100,
      feedbackCount: prevWeights.feedbackCount + 1,
      learningRate: prevWeights.learningRate
    };

    setRlWeights(updatedWeights);
    localStorage.setItem("deepguard_rl_weights", JSON.stringify(updatedWeights));

    const newEpisode: RLEpisode = {
      episode: updatedWeights.feedbackCount,
      reward: rewardValue,
      feedbackType: feedbackType.toUpperCase().replace("_", " "),
      previousWeights: prevWeights,
      newWeights: updatedWeights,
      timestamp: new Date().toLocaleTimeString()
    };

    const updatedHistory = [newEpisode, ...rlHistory];
    setRlHistory(updatedHistory);
    localStorage.setItem("deepguard_rl_history", JSON.stringify(updatedHistory));

    // Recalculate score live on report based on feedback
    if (analysisResult) {
      const oldScore = analysisResult.trust_score;
      let scoreAdjustment = Math.round((updatedWeights.realDocBias - prevWeights.realDocBias) + (updatedWeights.overallCalibration - prevWeights.overallCalibration));
      const adjustedScore = Math.min(100, Math.max(0, oldScore + scoreAdjustment));
      
      setAnalysisResult(prev => {
        if (!prev) return null;
        let newClassification = prev.classification;
        if (adjustedScore > 70) {
          newClassification = "Likely Real";
        } else if (adjustedScore > 40) {
          newClassification = "Suspicious";
        } else {
          newClassification = "Likely Fake";
        }

        return {
          ...prev,
          trust_score: adjustedScore,
          classification: newClassification as "Likely Real" | "Suspicious" | "Likely Fake",
          explanation: `Neural network layers adjusted! Applied reinforcement gradients (Reward: ${rewardValue > 0 ? "+" : ""}${rewardValue}, Episode ${updatedWeights.feedbackCount}). Trust calibrator updated score from ${oldScore}% to ${adjustedScore}%. ${prev.explanation}`
        };
      });

      // Update in scans array
      if (scans.length > 0 && currentFile) {
        setScans(prevScans => {
          const idx = prevScans.findIndex(s => s.fileName === currentFile.name);
          if (idx !== -1) {
            const updated = [...prevScans];
            let newClassification = "Suspicious";
            if (adjustedScore > 70) newClassification = "Likely Real";
            else if (adjustedScore <= 40) newClassification = "Likely Fake";

            updated[idx] = {
              ...updated[idx],
              trustScore: adjustedScore,
              classification: newClassification
            };
            localStorage.setItem("deepguard_scans", JSON.stringify(updated));
            return updated;
          }
          return prevScans;
        });
      }
    }

    toast.success(`Reinforcement optimization applied: Weights re-calculated (Episode #${updatedWeights.feedbackCount}).`);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 10 * 1024 * 1024) {
      toast.error("Payload too large. Max allowed size: 10MB");
      return;
    }

    const toastId = toast.loading("Uploading forensic evidence package...");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const uploadResponse = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error("Local filesystem upload failure");
      }

      const uploadResult = await uploadResponse.json();
      toast.dismiss(toastId);

      const fileReader = new FileReader();
      fileReader.onloadend = async () => {
        const base64Uri = fileReader.result as string;
        setCurrentFile({
          path: uploadResult.filePath,
          name: uploadResult.fileName,
          mimeType: uploadResult.mimeType,
          base64: base64Uri
        });
        await runAnalysisPipeline(base64Uri, uploadResult.mimeType, uploadResult.fileName, uploadResult.filePath);
      };
      
      fileReader.readAsDataURL(file);
    } catch (err: any) {
      toast.dismiss(toastId);
      console.error(err);
      toast.error("Evidence upload failed: " + (err.message || "Server issue"));
    }
  };

  const filteredScans = scans.filter(scan => {
    const matchesSearch = scan.fileName.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-sans">
      <Toaster position="top-right" theme="dark" />

      {/* FIXED SOLID OPAQUE NAVBAR */}
      <nav id="opaque-navbar" className="fixed top-0 left-0 w-full h-16 z-50 bg-slate-900 border-b border-slate-800 flex items-center px-6 justify-between">
        <div 
          className="flex items-center gap-2 cursor-pointer select-none"
          onClick={() => {
            if (user) setView("dashboard");
          }}
        >
          <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-500/30 flex items-center justify-center">
            <Shield className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <span className="font-bold tracking-tight text-white block">DEEPGUARD</span>
            <span className="text-[8px] font-mono text-cyan-400 font-bold uppercase tracking-widest block leading-none">FORENSIC SUITE</span>
          </div>
        </div>

        {user && (
          <div className="flex items-center gap-4">
            <div className="hidden md:flex flex-col items-end leading-none">
              <span className="text-[9px] font-mono text-cyan-400/70 uppercase">Agent Level</span>
              <span className="text-xs font-semibold text-slate-300">ADMINISTRATOR</span>
            </div>
            <button 
              onClick={handleLogout}
              className="flex items-center gap-1.5 h-8 px-3 rounded-md bg-slate-800 hover:bg-rose-900/40 hover:text-rose-400 text-slate-300 text-xs font-mono font-bold transition-all border border-slate-700 hover:border-rose-800/60"
            >
              <LogOut className="w-3.5 h-3.5" />
              <span>LOGOUT</span>
            </button>
          </div>
        )}
      </nav>

      {/* MAIN CONTAINER */}
      <main className="flex-grow pt-28 px-4 md:px-8 max-w-[1440px] mx-auto w-full pb-16">
        
        {/* STEP-BY-STEP SIMULATION / PROGRESS VIEW OVERLAY */}
        {loading && (
          <div className="space-y-8 animate-fade-in">
            <div className="text-center max-w-md mx-auto">
              <div className="inline-flex p-3 bg-cyan-950/40 rounded-full border border-cyan-500/20 mb-4 animate-pulse">
                <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              </div>
              <h2 className="text-xl font-bold tracking-tight text-cyan-400 uppercase font-mono">Forensic Scan Active</h2>
              <p className="text-xs text-slate-400 mt-1 font-mono uppercase tracking-widest">establishing secure digital audit pipeline</p>
            </div>

            {/* Overall progress indicator bar */}
            <div className="bg-slate-900/60 border border-slate-800 p-6 rounded-2xl max-w-lg mx-auto">
              <div className="flex justify-between text-xs font-mono text-slate-400 mb-2">
                <span>ESTIMATED TIME REMAINING</span>
                <span>{estimatedTime}s</span>
              </div>
              <div className="w-full bg-slate-950 rounded-full h-3 border border-slate-800 overflow-hidden">
                <div 
                  className="bg-gradient-to-r from-cyan-400 to-indigo-500 h-full rounded-full transition-all duration-350"
                  style={{ width: `${overallProgress}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-2">
                <span>PROGRESS INTEGRITY MATRIX</span>
                <span>{overallProgress}%</span>
              </div>
            </div>

            {/* Pipeline nodes */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 max-w-4xl mx-auto">
              {analysisSteps.map((step) => (
                <div 
                  key={step.id} 
                  className={cn(
                    "p-4 rounded-xl border text-left flex flex-col justify-between h-40 transition-all",
                    step.status === "loading" ? "bg-slate-900 border-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.15)] animate-pulse" :
                    step.status === "completed" ? "bg-slate-900/90 border-emerald-500/40 text-slate-300" :
                    "bg-slate-900/30 border-slate-800 opacity-40 text-slate-500"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono leading-none tracking-widest font-extrabold uppercase">{step.label}</span>
                    <div className={cn(
                      "w-6 h-6 rounded-full flex items-center justify-center",
                      step.status === "loading" ? "text-cyan-400" :
                      step.status === "completed" ? "text-emerald-400" : "text-slate-600"
                    )}>
                      {step.status === "loading" ? <Loader2 className="w-4 h-4 animate-spin" /> :
                       step.status === "completed" ? <CheckCircle className="w-4 h-4" /> : step.icon}
                    </div>
                  </div>
                  <div>
                    <p className="text-[11px] font-semibold tracking-tight text-white mb-1 line-clamp-2 leading-tight">
                      {step.status === "loading" ? `ACTIVE: ${activeSubtask}` : step.description}
                    </p>
                    <span className="text-[9px] font-mono uppercase text-slate-400">
                      {step.status === "loading" ? "Analyzing sectors..." : 
                       step.status === "completed" ? "VERIFIED" : "AWAITING..."}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Kernel Live Stream logs */}
            {processLogs.length > 0 && (
              <div className="max-w-2xl mx-auto space-y-2 text-left">
                <span className="text-[10px] font-mono text-cyan-400 font-bold tracking-widest block uppercase">Kernel Live Log Stream</span>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 h-44 overflow-y-auto font-mono text-[10px] space-y-1.5 scrollbar-thin">
                  {processLogs.map((log) => (
                    <div key={log.id} className="flex gap-2">
                      <span className="text-slate-500">[{log.timestamp}]</span>
                      <span className={cn(
                        "font-bold uppercase",
                        log.type === "success" ? "text-emerald-400" :
                        log.type === "running" ? "text-cyan-400" : "text-indigo-400"
                      )}>
                        {log.processName ? `[${log.processName}]` : "[SYS]"}
                      </span>
                      <span className="text-slate-300 font-mono">{log.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* VIEW 1: LOGIN CARD SCREEN */}
        {view === "login" && !loading && (
          <div className="max-w-md mx-auto my-12 animate-fade-in">
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-[3px] bg-gradient-to-r from-cyan-500 via-indigo-500 to-rose-500" />
              
              <div className="text-center mb-8">
                <div className="w-16 h-16 bg-slate-800 border-2 border-slate-700/60 rounded-2xl flex items-center justify-center mx-auto mb-4 text-cyan-400 shadow-[0_0_20px_rgba(6,182,212,0.1)]">
                  <Lock className="w-8 h-8" />
                </div>
                <h2 className="text-2xl font-bold font-mono tracking-tight text-white uppercase">Agent Access</h2>
                <p className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest mt-1">Biometric Verification Required</p>
              </div>

              <form onSubmit={handleLogin} className="space-y-6">
                <div className="space-y-2">
                  <label className="text-[10px] font-mono uppercase text-slate-400 tracking-wider block">Agent Credentials</label>
                  <div className="relative">
                    <input 
                      type="email" 
                      placeholder="ENTER REGISTERED EMAIL" 
                      value={email}
                      onChange={(e) => {
                        setEmail(e.target.value);
                        if (emailError) setEmailError("");
                      }}
                      required
                      className={cn(
                        "w-full h-12 bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 text-xs font-mono uppercase tracking-wider text-white focus:outline-none focus:border-cyan-500/80 transition-all",
                        emailError && "border-rose-500"
                      )}
                    />
                    <UserIcon className="absolute left-3.5 top-3.5 w-4 h-4 text-slate-500" />
                  </div>
                  {emailError && <p className="text-[9px] font-mono text-rose-500 tracking-wider block uppercase mt-1">{emailError}</p>}
                </div>

                <div className="space-y-2">
                  <label className="text-[10px] font-mono uppercase text-slate-400 tracking-wider block">Encryption Key Matrix</label>
                  <div className="relative">
                    <input 
                      type="password" 
                      placeholder="••••••••" 
                      value={password}
                      onChange={(e) => {
                        setPassword(e.target.value);
                        if (passwordError) setPasswordError("");
                      }}
                      required
                      className={cn(
                        "w-full h-12 bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 text-xs font-mono text-white focus:outline-none focus:border-cyan-500/80 transition-all",
                        passwordError && "border-rose-500"
                      )}
                    />
                    <Key className="absolute left-3.5 top-3.5 w-4 h-4 text-slate-500" />
                  </div>
                  {passwordError && <p className="text-[9px] font-mono text-rose-500 tracking-wider block uppercase mt-1">{passwordError}</p>}
                </div>

                <button 
                  type="submit"
                  className="w-full h-12 bg-cyan-500 hover:bg-cyan-600 active:bg-cyan-700 text-slate-950 font-bold tracking-widest text-xs uppercase rounded-xl shadow-[0_0_20px_rgba(6,182,212,0.2)] transition-all flex items-center justify-center gap-2 select-none"
                >
                  <Shield className="w-4 h-4" />
                  <span>Authorize Uplink</span>
                </button>
              </form>

              <div className="mt-8 pt-6 border-t border-slate-800 text-center">
                <span className="text-[8px] font-mono text-slate-600 uppercase tracking-widest block">Authorized Forensic Personnel Only</span>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 2: DASHBOARD VIEW */}
        {view === "dashboard" && !loading && (
          <div className="space-y-8 animate-fade-in">
            {/* Main Title Section */}
            <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-white font-mono flex items-center gap-2">
              <span className="w-1.5 h-6 bg-cyan-400 rounded-full inline-block" />
              INITIALIZE FORENSIC SCAN
            </h2>

            {/* Upper grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              
              {/* Left pane: File upload + RL calibration side-by-side or stacked on mobile */}
              <div className="lg:col-span-8 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
                  
                  {/* Forensic file upload input sector */}
                  <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 relative overflow-hidden flex flex-col justify-between min-h-[350px]">
                    <div className="flex flex-col items-center justify-center text-center my-auto">
                      <div className="w-14 h-14 bg-cyan-950/20 rounded-2xl border border-cyan-500/30 flex items-center justify-center text-cyan-400 mb-4 shrink-0">
                        <Upload className="w-7 h-7" />
                      </div>
                      <p className="text-sm text-slate-200 font-bold mb-1.5">Upload Evidence Document</p>
                      <p className="text-[11px] text-slate-400 max-w-[240px] mx-auto mb-5 leading-relaxed">
                        Accepts PDF, PNG, or JPEG records. Our neural analyzer pipeline audits layout markers and signature models for synthetic tampering.
                      </p>

                      <label className="inline-flex items-center justify-center h-10 px-5 rounded-xl bg-cyan-500 hover:bg-cyan-600 text-slate-950 font-bold text-xs tracking-wider uppercase cursor-pointer select-none shadow-[0_0_12px_rgba(6,182,212,0.15)] transition-all">
                        <span>Payload Selection</span>
                        <input 
                          type="file" 
                          className="hidden" 
                          accept="image/*,application/pdf"
                          onChange={handleFileUpload}
                        />
                      </label>
                      <span className="text-[9px] font-mono text-slate-500 uppercase mt-3">max upload size: 10mb</span>
                    </div>
                  </div>

                  {/* Reinforcement Calibration Dashboard card */}
                  <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 flex flex-col justify-between min-h-[350px]">
                    <div>
                      <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-800">
                        <span className="text-xs font-mono text-cyan-400 tracking-wider font-bold uppercase">Reinforcement Calibration (RL)</span>
                        <Activity className="w-3.5 h-3.5 text-indigo-400 animate-pulse" />
                      </div>
                      <div className="space-y-2.5 font-mono text-[11px]">
                        <div className="flex justify-between items-center bg-slate-950/40 p-1.5 rounded border border-slate-800/40">
                          <span className="text-slate-400 text-[10px]">GENUINE BIAS</span>
                          <span className="text-emerald-400 font-bold">+{rlWeights.realDocBias} pts</span>
                        </div>
                        <div className="flex justify-between items-center bg-slate-950/40 p-1.5 rounded border border-slate-800/40">
                          <span className="text-slate-400 text-[10px]">HARSH MULTIPLIER</span>
                          <span className="text-cyan-400 font-bold">{rlWeights.strictnessModifier}x</span>
                        </div>
                        <div className="flex justify-between items-center bg-slate-950/40 p-1.5 rounded border border-slate-800/40">
                          <span className="text-slate-400 text-[10px]">BASE CALIBRATOR</span>
                          <span className="text-indigo-400 font-bold">+{rlWeights.overallCalibration} pts</span>
                        </div>
                        <div className="flex justify-between items-center px-1">
                          <span className="text-slate-500 text-[9px]">TOTAL EPISODES</span>
                          <span className="text-slate-300 font-bold text-[9px]">{rlWeights.feedbackCount} epochs</span>
                        </div>
                      </div>
                    </div>
                    
                    {rlHistory.length > 0 ? (
                      <div className="mt-3 pt-2.5 border-t border-slate-800/70">
                        <span className="text-[8px] text-slate-500 tracking-widest block mb-1.5 uppercase">Feedback Gradient Steps</span>
                        <div className="space-y-1 text-[9px] font-mono max-h-[75px] overflow-y-auto scrollbar-thin">
                          {rlHistory.slice(0, 3).map((hist, index) => (
                            <div key={index} className="flex justify-between items-center text-slate-400 bg-slate-950/30 px-2 py-0.5 rounded">
                              <span className="text-slate-350 truncate max-w-[120px]">EP {hist.episode}: {hist.feedbackType}</span>
                              <span className={cn(
                                "font-bold",
                                hist.reward >= 0 ? "text-emerald-400" : "text-rose-400"
                              )}>
                                {hist.reward >= 0 ? "+" : ""}{hist.reward} R
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="mt-3 pt-2 border-t border-slate-800/70 text-center py-2">
                        <span className="text-[9px] font-mono text-slate-600 uppercase">Awaiting training reward feedback</span>
                      </div>
                    )}
                  </div>

                </div>
              </div>

              {/* Right column: System info panels */}
              <div className="lg:col-span-4 space-y-6">
                
                {/* System integrity list */}
                <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 min-h-[162px] flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
                      <span className="text-xs font-mono text-cyan-400 tracking-wider font-bold">System Integrity</span>
                      <Activity className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                    </div>
                    <div className="space-y-3 font-mono text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-400">NEURAL CODES</span>
                        <span className="text-emerald-400 font-bold">ONLINE</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">METADATA ENGINE</span>
                        <span className="text-emerald-400 font-bold">SECURED</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">ENCRYPTION LEVEL</span>
                        <span className="text-indigo-400 font-bold">AES-256</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">DEEP MODEL</span>
                        <span className="text-slate-300">v3-FLASH</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Evidence count metrics log overlay */}
                <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 min-h-[162px] flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
                      <span className="text-xs font-mono text-cyan-400 tracking-wider font-bold">Operational Logs</span>
                      <History className="w-3.5 h-3.5 text-cyan-400" />
                    </div>
                    {scans.length > 0 ? (
                      <div className="space-y-2.5 font-mono text-[10px] max-h-24 overflow-y-auto scrollbar-thin">
                        {scans.slice(0, 3).map((scan) => (
                          <div key={scan.id} className="p-2 bg-slate-950/40 rounded border border-slate-800/60 flex items-center justify-between">
                            <span className="text-slate-300 truncate max-w-[124px] uppercase">{scan.fileName}</span>
                            <span className={cn(
                              "font-bold",
                              scan.trustScore > 70 ? "text-emerald-400" :
                              scan.trustScore > 40 ? "text-cyan-400" : "text-rose-400"
                            )}>
                              {scan.classification.toUpperCase()}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-4">
                        <span className="text-[10px] font-mono text-slate-600 uppercase">Archive history empty</span>
                      </div>
                    )}
                  </div>
                </div>

              </div>

            </div>

            {/* LOWER PORTION: Evidence Archive List */}
            <div className="space-y-4">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-1.5 h-6 bg-cyan-400 rounded-full" />
                  <h3 className="text-xl font-bold font-mono tracking-tight text-white uppercase">Historical Evidence Archive</h3>
                  <span className="bg-slate-800 border border-slate-700 px-2 py-0.5 text-[9px] font-mono text-slate-400 rounded">{scans.length} RECORDS</span>
                </div>

                <div className="flex items-center gap-2">
                  <div className="relative">
                    <input 
                      type="text" 
                      placeholder="SEARCH RECORDS..." 
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="h-10 w-52 bg-slate-900 border border-slate-800 rounded-xl pl-9 pr-3 text-xs font-mono uppercase text-white focus:outline-none focus:border-cyan-500/80 transition-all placeholder:text-slate-500"
                    />
                    <Search className="absolute left-3 top-3.5 w-3.5 h-3.5 text-slate-500" />
                  </div>
                  {scans.length > 0 && (
                    <button 
                      onClick={clearArchive}
                      className="h-10 px-4 rounded-xl border border-rose-500/20 hover:border-rose-500 hover:bg-rose-900/10 text-rose-500 hover:text-white font-mono text-xs font-black tracking-widest transition-all uppercase flex items-center gap-1.5"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      <span>PURGE MATRIX</span>
                    </button>
                  )}
                </div>
              </div>

              {filteredScans.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {filteredScans.map((scan) => {
                    const statusColor = scan.trustScore > 70 ? "text-emerald-400 bg-emerald-950/20 border-emerald-500/30" : 
                                      scan.trustScore > 40 ? "text-cyan-400 bg-cyan-950/20 border-cyan-500/30" : 
                                      "text-rose-400 bg-rose-950/20 border-rose-500/30";
                    const scoreColor = scan.trustScore > 70 ? "text-emerald-400" : 
                                     scan.trustScore > 40 ? "text-cyan-400" : 
                                     "text-rose-400";
                    return (
                      <div 
                        key={scan.id}
                        className="bg-slate-900 border border-slate-800/80 rounded-2xl p-5 hover:border-cyan-500/40 cursor-pointer transition-all flex flex-col justify-between group relative"
                        onClick={() => {
                          if (currentFile && currentFile.name === scan.fileName) {
                            setView("analysis");
                          } else {
                            toast.info("Select payload input matching this file to view forensic detail report.");
                          }
                        }}
                      >
                        <div className="space-y-4">
                          <div className="flex items-start justify-between">
                            <div className="w-10 h-10 bg-slate-800 rounded-lg flex items-center justify-center text-slate-400 group-hover:text-cyan-400 transition-colors">
                              <FileText className="w-5 h-5" />
                            </div>
                            <button 
                              onClick={(e) => {
                                e.stopPropagation();
                                deleteScan(scan.id);
                              }}
                              className="w-8 h-8 rounded-lg bg-slate-950 hover:bg-rose-950/40 text-slate-500 hover:text-rose-400 border border-slate-800 flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
                              title="Delete Record"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                          <div>
                            <p className="text-sm font-bold tracking-tight text-white uppercase truncate" title={scan.fileName}>
                              {scan.fileName}
                            </p>
                            <p className="text-[9px] font-mono text-slate-500 mt-1 uppercase">
                              {new Date(scan.timestamp).toLocaleString()}
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center justify-between pt-4 mt-4 border-t border-slate-850">
                          <span className={cn("text-[9px] font-mono font-black uppercase tracking-widest px-2.5 py-0.5 rounded border", statusColor)}>
                            {scan.classification}
                          </span>
                          <span className={cn("text-xs font-mono font-bold", scoreColor)}>
                            {scan.trustScore}% TRUST
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="py-20 text-center bg-slate-900/40 border-2 border-dashed border-slate-800 rounded-3xl">
                  <div className="w-12 h-12 bg-slate-900 border border-slate-800 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-600">
                    <Search className="w-6 h-6" />
                  </div>
                  <h4 className="text-sm font-bold text-slate-400 font-mono uppercase tracking-widest">No matching report log entries</h4>
                  <p className="text-xs text-slate-500">Database query returned empty set.</p>
                </div>
              )}
            </div>

          </div>
        )}

        {/* VIEW 3: ANALYSIS/REPORT RESULTS VIEW */}
        {view === "analysis" && currentFile && analysisResult && !loading && (
          <div className="space-y-8 animate-fade-in text-left">
            
            {/* Analysis Header Toolbar */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <button 
                  onClick={() => setView("dashboard")}
                  className="flex items-center justify-center w-10 h-10 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition-all active:scale-95"
                  title="Return to Dashboard"
                >
                  <ChevronRight className="w-5 h-5 rotate-180" />
                </button>
                <div>
                  <h2 className="text-2xl font-bold font-mono text-white tracking-tight flex items-center gap-2">
                    FORENSIC REPORT
                    <span className="text-[10px] bg-cyan-950/60 border border-cyan-500/20 px-2 py-0.5 text-cyan-400 font-mono rounded">
                      VERIFIED
                    </span>
                  </h2>
                  <p className="text-[10px] font-mono text-slate-400 uppercase tracking-widest mt-1">UPLINK IDENTIFIER: {currentFile.name.toUpperCase()}</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button 
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  className={cn(
                    "h-10 px-4 rounded-xl border font-mono font-bold text-xs uppercase tracking-wider transition-all flex items-center gap-2 select-none",
                    showHeatmap ? "bg-cyan-500 border-cyan-500 text-slate-950 shadow-[0_0_15px_rgba(6,182,212,0.15)]" : "bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700"
                  )}
                >
                  {showHeatmap ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  <span>{showHeatmap ? "Mute Overlay" : "Draw Overlay"}</span>
                </button>
                
                <button 
                  onClick={() => runAnalysisPipeline(currentFile.base64, currentFile.mimeType, currentFile.name, currentFile.path)}
                  className="h-10 px-4 rounded-xl bg-slate-905 border border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-400 font-mono text-xs font-bold transition-all uppercase flex items-center gap-1.5 whitespace-nowrap"
                >
                  <RefreshCw className="w-3.5 h-3.5 animate-spin-reverse" />
                  <span>Re_Analyze</span>
                </button>
              </div>
            </div>

            {/* Main view structures */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              
              {/* Left pane: file image preview canvas & charts */}
              <div className="lg:col-span-8 space-y-8 lg:sticky lg:top-24 max-h-[calc(100vh-8rem)] lg:overflow-y-auto scrollbar-none">
                
                {/* Visual File Preview Box */}
                <div className="bg-slate-900 border border-slate-800 rounded-3xl p-4 relative overflow-hidden flex items-center justify-center min-h-[380px] shadow-inner">
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent z-10 pointer-events-none" />
                  
                  {currentFile.mimeType.includes("pdf") ? (
                    <div className="w-full h-[450px] relative z-0 bg-slate-950 rounded-2xl overflow-hidden self-stretch flex items-center justify-center">
                      <object
                        data={currentFile.base64}
                        type="application/pdf"
                        className="w-full h-full"
                      >
                        <iframe 
                          src={`${currentFile.base64}#toolbar=0&navpanes=0&scrollbar=0`}
                          className="w-full h-full border-none"
                          title="PDF Forensic Copy"
                        />
                      </object>
                      <HeatmapOverlay regions={analysisResult.heatmap_regions} show={showHeatmap} />
                    </div>
                  ) : (
                    <div className="relative z-0 max-w-full overflow-hidden rounded-2xl">
                      <img 
                        src={currentFile.base64} 
                        alt="Evidence Visual Capture"
                        className="max-h-[500px] w-auto object-contain rounded-2xl"
                      />
                      <HeatmapOverlay regions={analysisResult.heatmap_regions} show={showHeatmap} />
                    </div>
                  )}

                  {/* Document caption footer overlay */}
                  <div className="absolute bottom-4 left-4 right-4 z-20 flex justify-between items-end">
                    <div>
                      <span className="text-[8px] font-mono text-cyan-400 uppercase tracking-widest font-black block">Payload Capture</span>
                      <span className="text-sm font-bold text-white block truncate uppercase max-w-[260px]">{currentFile.name}</span>
                    </div>
                    <span className="bg-slate-950/80 backdrop-blur-md border border-slate-800 text-[9px] font-mono text-slate-400 px-3 py-1 rounded-lg">
                      {currentFile.mimeType.toUpperCase()}
                    </span>
                  </div>
                </div>

                {/* Accuracy vector graph panel */}
                <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 space-y-6">
                  <div>
                    <h3 className="text-lg font-bold font-mono uppercase text-white tracking-tight">Forensic Accuracy Metrics</h3>
                    <p className="text-[10px] text-slate-400 font-mono uppercase tracking-widest mt-1">Multi-vector neural check results</p>
                  </div>
                  <ForensicBarChart score={analysisResult.trust_score} />
                </div>

              </div>

              {/* Right pane: score, verdict, finding list & suggest tabs */}
              <div className="lg:col-span-4 space-y-8">
                
                {/* Score panel */}
                <div className={cn(
                  "bg-slate-900 border rounded-3xl p-6 relative overflow-hidden text-center shadow-lg transition-colors",
                  analysisResult.trust_score > 70 ? "border-emerald-500/20" :
                  analysisResult.trust_score > 40 ? "border-cyan-500/20" :
                  "border-rose-500/20"
                )}>
                  <div className="absolute top-0 right-0 p-2 opacity-10">
                    <Shield className="w-16 h-16 text-white" />
                  </div>
                  
                  <div className="mb-6">
                    <span className={cn(
                      "inline-block text-[10px] font-mono font-black uppercase tracking-[0.25em] px-4 py-1 rounded-full border mb-6",
                      analysisResult.trust_score > 70 ? "text-emerald-400 border-emerald-550/30 bg-emerald-950/20" :
                      analysisResult.trust_score > 40 ? "text-cyan-400 border-cyan-550/30 bg-cyan-950/20" :
                      "text-rose-400 border-rose-550/30 bg-rose-950/20"
                    )}>
                      {analysisResult.classification.toUpperCase()}
                    </span>
                    <ForensicScoreGauge score={analysisResult.trust_score} />
                  </div>
                  <div className="border-t border-slate-800/80 pt-4 text-left">
                    <span className="text-[8px] font-mono text-slate-500 uppercase tracking-widest block mb-1">Synthesizer analysis conclusion:</span>
                    <p className="text-xs text-slate-400 leading-relaxed italic">
                      "{analysisResult.explanation}"
                    </p>
                  </div>
                </div>

                {/* REINFORCEMENT LEARNING FEEDBACK & REWARD ADAPTATION */}
                <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
                  <div>
                    <span className="text-[8px] font-mono text-cyan-400 font-bold tracking-widest block uppercase">Human-In-The-Loop Training</span>
                    <h4 className="text-sm font-bold font-mono text-white tracking-tight uppercase mt-0.5">Reward Model Prediction</h4>
                    <p className="text-[10px] text-slate-400 leading-normal mt-1">
                      Report accuracy dictates reinforcement parameters. Assigning rewards corrects weights and automatically updates the analyzer dynamically.
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => applyRLFeedback("excellent", 10)}
                      className="bg-emerald-950/40 border border-emerald-500/20 hover:border-emerald-400/50 p-3 rounded-xl text-left transition-all active:scale-95 group"
                    >
                      <div className="text-xs font-bold text-emerald-400 flex items-center justify-between mb-1">
                        <span>Excellent</span>
                        <span className="text-[9px] font-mono text-slate-500 group-hover:text-emerald-300 transition-colors">+10 R</span>
                      </div>
                      <p className="text-[9px] text-slate-400 leading-none">Perfect score & classification</p>
                    </button>

                    <button
                      onClick={() => applyRLFeedback("underestimated", 5)}
                      className="bg-cyan-950/40 border border-cyan-500/20 hover:border-cyan-400/50 p-3 rounded-xl text-left transition-all active:scale-95 group"
                    >
                      <div className="text-xs font-bold text-cyan-400 flex items-center justify-between mb-1">
                        <span>Too Strict</span>
                        <span className="text-[9px] font-mono text-slate-300 group-hover:text-cyan-300 transition-colors">+5 R</span>
                      </div>
                      <p className="text-[9px] text-slate-400 leading-none">Score too strict on genuine scan</p>
                    </button>

                    <button
                      onClick={() => applyRLFeedback("harsh", -3)}
                      className="bg-slate-950/30 border border-slate-800 hover:border-slate-500 p-3 rounded-xl text-left transition-all active:scale-95 group"
                    >
                      <div className="text-xs font-bold text-slate-300 flex items-center justify-between mb-1">
                        <span>Harsh Bias</span>
                        <span className="text-[9px] font-mono text-slate-500 group-hover:text-white transition-colors">-3 P</span>
                      </div>
                      <p className="text-[9px] text-slate-400 leading-none">Verdict is right, but score harsh</p>
                    </button>

                    <button
                      onClick={() => applyRLFeedback("false_positive", -10)}
                      className="bg-rose-950/40 border border-rose-500/20 hover:border-rose-400/50 p-3 rounded-xl text-left transition-all active:scale-95 group"
                    >
                      <div className="text-xs font-bold text-rose-400 flex items-center justify-between mb-1">
                        <span>Missed Leak</span>
                        <span className="text-[9px] font-mono text-slate-500 group-hover:text-rose-300 transition-colors">-10 P</span>
                      </div>
                      <p className="text-[9px] text-slate-400 leading-none">Failed to flag synthetic tampering</p>
                    </button>
                  </div>

                  <div className="pt-2">
                    <div className="flex justify-between items-center text-[9px] font-mono text-slate-500 mb-1">
                      <span>POLICY GRADIENT EPISODES</span>
                      <span>LR: {rlWeights.learningRate}</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                      <div className="h-full bg-cyan-400 transition-all rounded-full" style={{ width: `${Math.min(100, Math.max(10, rlWeights.feedbackCount * 20))}%` }} />
                    </div>
                  </div>
                </div>

                {/* Tabs panel */}
                <div className="space-y-4">
                  <div className="flex bg-slate-900 border border-slate-800 p-1.5 rounded-2xl">
                    <button 
                      onClick={() => setActiveTab("findings")}
                      className={cn(
                        "flex-1 py-2 text-xs font-mono font-bold uppercase tracking-wider rounded-xl transition-all",
                        activeTab === "findings" ? "bg-cyan-500 text-slate-950 shadow" : "text-slate-400 hover:text-white"
                      )}
                    >
                      Aura Findings
                    </button>
                    <button 
                      onClick={() => setActiveTab("suggestions")}
                      className={cn(
                        "flex-1 py-2 text-xs font-mono font-bold uppercase tracking-wider rounded-xl transition-all",
                        activeTab === "suggestions" ? "bg-cyan-500 text-slate-950 shadow" : "text-slate-400 hover:text-white"
                      )}
                    >
                      Protocols
                    </button>
                  </div>

                  <div className="space-y-3">
                    {activeTab === "findings" ? (
                      analysisResult.reasons.map((reason, i) => (
                        <div key={i} className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex items-start gap-3">
                          <div className="w-6 h-6 rounded-lg bg-rose-950 border border-rose-800 text-rose-400 flex items-center justify-center shrink-0 mt-0.5 text-[11px] font-bold">
                            !
                          </div>
                          <p className="text-xs text-slate-300 leading-normal font-medium">{reason}</p>
                        </div>
                      ))
                    ) : (
                      analysisResult.suggestions.map((suggest, i) => (
                        <div key={i} className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex items-start gap-3">
                          <div className="w-6 h-6 rounded-lg bg-cyan-950 border border-cyan-800 text-cyan-400 flex items-center justify-center shrink-0 mt-0.5 text-[11px] font-bold">
                            ✓
                          </div>
                          <p className="text-xs text-slate-300 leading-normal font-medium">{suggest}</p>
                        </div>
                      ))
                    )}
                  </div>
                </div>

              </div>

            </div>

            {/* Print & Export actions */}
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <Printer className="w-5 h-5 text-slate-400" />
                <div>
                  <span className="text-xs font-bold text-white block uppercase">Audit Output Matrix Setup</span>
                  <p className="text-[10px] text-slate-500 leading-tight block uppercase">Generates paper-evidence files directly to device memory</p>
                </div>
              </div>
              <div className="flex gap-2 w-full md:w-auto">
                <button 
                  onClick={() => window.print()}
                  className="flex-grow md:flex-none h-11 px-5 rounded-xl border border-slate-700/60 hover:border-slate-500 text-white font-mono text-xs font-bold transition-all uppercase flex items-center justify-center gap-1.5"
                >
                  <Download className="w-4 h-4" />
                  <span>EXPORT REPORT</span>
                </button>
                <button 
                  onClick={() => setView("dashboard")}
                  className="flex-grow md:flex-none h-11 px-5 rounded-xl bg-cyan-500 hover:bg-cyan-600 text-slate-950 font-bold text-xs uppercase transition-all select-none shadow-[0_0_15px_rgba(6,182,212,0.1)] flex items-center justify-center"
                >
                  <span>New Scan</span>
                </button>
              </div>
            </div>

          </div>
        )}

      </main>

      {/* SYSTEM TELEMETRY FOOTER */}
      <footer className="mt-auto bg-slate-905 border-t border-slate-800 py-3 px-6 text-xs text-slate-400 font-mono flex justify-between items-center bg-slate-900">
        <div className="flex items-center gap-2">
          <Shield className="w-4 h-4 text-cyan-400" />
          <span className="text-[10px] tracking-tight text-slate-300">DEEPGUARD // SECURE INTEL MATRIX SECURITY PROTOCOL</span>
        </div>
        <div className="flex gap-6 text-[11px]">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-bold uppercase">LATENCY:</span>
            <span className="text-cyan-400">0.0014ms</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-bold uppercase">NODES:</span>
            <span className="text-slate-300">128/128</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 font-bold uppercase">STATUS:</span>
            <span className="text-emerald-400 animate-pulse font-bold">OPERATIONAL</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
