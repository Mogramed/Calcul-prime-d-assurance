export type UserRole = "customer" | "admin";

export type ApiErrorBody = {
  detail?: string | Array<Record<string, unknown>>;
  request_id?: string;
};

export type SessionUser = {
  id: string;
  created_at_utc: string;
  email: string;
  role: UserRole;
  is_active: boolean;
};

export type AuthSessionResponse = {
  authenticated: boolean;
  user: SessionUser | null;
  session_token?: string | null;
  expires_at_utc?: string | null;
};

export type AuthCredentialsInput = {
  email: string;
  password: string;
};

export type AdminUserSummaryResponse = SessionUser;

export type AdminUserListResponse = {
  count: number;
  users: AdminUserSummaryResponse[];
};

export type AdminQuoteSummaryResponse = {
  id: string;
  created_at_utc: string;
  run_id: string;
  type_contrat: string;
  marque_vehicule: string;
  modele_vehicule: string;
  prime_prediction: number;
  user_id: string | null;
  owner_email: string | null;
  deleted_at_utc: string | null;
};

export type AdminQuoteListResponse = {
  count: number;
  quotes: AdminQuoteSummaryResponse[];
};
