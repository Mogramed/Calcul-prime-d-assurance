export interface PredictionInput {
  bonus: number;
  type_contrat: string;
  duree_contrat: number;
  anciennete_info: number;
  freq_paiement: string;
  paiement: string;
  utilisation: string;
  code_postal: number | string;
  conducteur2: string;
  age_conducteur1: number;
  age_conducteur2: number;
  sex_conducteur1: string;
  sex_conducteur2: string;
  anciennete_permis1: number;
  anciennete_permis2: number;
  anciennete_vehicule: number;
  cylindre_vehicule: number;
  din_vehicule: number;
  essence_vehicule: string;
  marque_vehicule: string;
  modele_vehicule: string;
  debut_vente_vehicule: number;
  fin_vente_vehicule: number;
  vitesse_vehicule: number;
  type_vehicule: string;
  prix_vehicule: number;
  poids_vehicule: number;
}

export interface QuoteResultResponse {
  frequency_prediction: number;
  severity_prediction: number;
  prime_prediction: number;
}

export interface QuoteEmailDeliveryResponse {
  status: "sent" | "failed" | "skipped";
  recipient_email?: string | null;
}

export interface QuoteResponse {
  id: string;
  created_at_utc: string;
  run_id: string;
  input_payload: PredictionInput;
  result: QuoteResultResponse;
  email_delivery?: QuoteEmailDeliveryResponse | null;
}

export interface QuoteSummaryResponse {
  id: string;
  created_at_utc: string;
  run_id: string;
  type_contrat: string;
  marque_vehicule: string;
  modele_vehicule: string;
  prime_prediction: number;
}

export interface QuoteListResponse {
  count: number;
  quotes: QuoteSummaryResponse[];
}

export interface ApiErrorResponse {
  detail: string | Array<Record<string, unknown>>;
  request_id?: string;
}
