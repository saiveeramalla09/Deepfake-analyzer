export interface User {
  id: string;
  email: string;
  name: string;
}

export interface ScanRecord {
  id: string;
  fileName: string;
  filePath: string;
  timestamp: string;
  trustScore: number;
  classification: string;
}
