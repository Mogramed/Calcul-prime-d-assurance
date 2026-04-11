import { z } from "zod";

import type { PredictionInput } from "@/generated/client/types.gen";
import {
  brandOptions,
  contractOptions,
  fuelOptions,
  genderOptions,
  getChoiceLabel,
  getModelOptions,
  paymentFrequencyOptions,
  paymentStatusOptions,
  secondDriverOptions,
  type QuoteOption,
  usageOptions,
  vehicleTypeOptions,
} from "@/lib/quote-catalog";
import { formatCurrency, formatInteger, formatTitleCase } from "@/lib/format";

const requiredText = z.string().trim().min(1, "Champ requis");
const requiredInteger = z
  .string()
  .trim()
  .min(1, "Champ requis")
  .refine((value) => /^-?\d+$/.test(value), "Entier requis");
const requiredNumber = z
  .string()
  .trim()
  .min(1, "Champ requis")
  .refine((value) => !Number.isNaN(Number(value)), "Nombre requis");
const optionalInteger = z
  .string()
  .trim()
  .refine((value) => value === "" || /^-?\d+$/.test(value), "Entier invalide");

export const quoteFormSchema = z
  .object({
    bonus: requiredNumber,
    type_contrat: requiredText,
    duree_contrat: requiredInteger,
    anciennete_info: requiredInteger,
    freq_paiement: requiredText,
    paiement: requiredText,
    utilisation: requiredText,
    code_postal: z
      .string()
      .trim()
      .min(1, "Champ requis")
      .refine((value) => /^\d{4,5}$/.test(value), "Code postal invalide"),
    conducteur2: requiredText,
    age_conducteur1: requiredInteger,
    age_conducteur2: optionalInteger,
    sex_conducteur1: requiredText,
    sex_conducteur2: z.string().trim(),
    anciennete_permis1: requiredInteger,
    anciennete_permis2: optionalInteger,
    anciennete_vehicule: requiredNumber,
    cylindre_vehicule: requiredInteger,
    din_vehicule: requiredInteger,
    essence_vehicule: requiredText,
    marque_vehicule: requiredText,
    modele_vehicule: requiredText,
    debut_vente_vehicule: requiredInteger,
    fin_vente_vehicule: requiredInteger,
    vitesse_vehicule: requiredInteger,
    type_vehicule: requiredText,
    prix_vehicule: requiredInteger,
    poids_vehicule: requiredInteger,
  })
  .superRefine((values, context) => {
    if (values.conducteur2 !== "Yes") {
      return;
    }

    if (!values.age_conducteur2.trim()) {
      context.addIssue({
        code: "custom",
        path: ["age_conducteur2"],
        message: "Champ requis",
      });
    }

    if (!values.sex_conducteur2.trim()) {
      context.addIssue({
        code: "custom",
        path: ["sex_conducteur2"],
        message: "Choisissez une option",
      });
    }

    if (!values.anciennete_permis2.trim()) {
      context.addIssue({
        code: "custom",
        path: ["anciennete_permis2"],
        message: "Champ requis",
      });
    }
  });

export type QuoteFormValues = z.infer<typeof quoteFormSchema>;
export type QuoteFieldName = keyof QuoteFormValues;
export type QuoteStepId = "profile" | "drivers" | "vehicle";

type FieldKind = "input" | "select";

export type QuoteFieldConfig = {
  name: QuoteFieldName;
  label: string;
  description: string;
  kind: FieldKind;
  placeholder?: string;
  inputMode?: "text" | "decimal" | "numeric";
  options?: readonly QuoteOption[];
};

export type QuoteStep = {
  id: QuoteStepId;
  eyebrow: string;
  title: string;
  description: string;
  fields: QuoteFieldName[];
};

export const defaultQuoteValues: QuoteFormValues = {
  bonus: "",
  type_contrat: "",
  duree_contrat: "",
  anciennete_info: "",
  freq_paiement: "",
  paiement: "",
  utilisation: "",
  code_postal: "",
  conducteur2: "",
  age_conducteur1: "",
  age_conducteur2: "",
  sex_conducteur1: "",
  sex_conducteur2: "",
  anciennete_permis1: "",
  anciennete_permis2: "",
  anciennete_vehicule: "",
  cylindre_vehicule: "",
  din_vehicule: "",
  essence_vehicule: "",
  marque_vehicule: "",
  modele_vehicule: "",
  debut_vente_vehicule: "",
  fin_vente_vehicule: "",
  vitesse_vehicule: "",
  type_vehicule: "",
  prix_vehicule: "",
  poids_vehicule: "",
};

export const quoteFieldConfigs: Record<QuoteFieldName, QuoteFieldConfig> = {
  bonus: {
    name: "bonus",
    label: "Bonus-malus",
    description: "Indiquez votre coefficient actuel.",
    kind: "input",
    inputMode: "decimal",
    placeholder: "0.58",
  },
  type_contrat: {
    name: "type_contrat",
    label: "Formule souhaitee",
    description: "Choisissez le niveau de couverture souhaite.",
    kind: "select",
    options: contractOptions,
  },
  duree_contrat: {
    name: "duree_contrat",
    label: "Duree du contrat",
    description: "Renseignez la duree prevue en annees.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "1",
  },
  anciennete_info: {
    name: "anciennete_info",
    label: "Anciennete du dossier",
    description: "Nombre d'annees d'historique disponible pour ce contrat.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "1",
  },
  freq_paiement: {
    name: "freq_paiement",
    label: "Frequence de paiement",
    description: "Choisissez votre rythme de paiement prefere.",
    kind: "select",
    options: paymentFrequencyOptions,
  },
  paiement: {
    name: "paiement",
    label: "Situation de paiement",
    description: "Selectionnez la situation la plus proche de votre contrat.",
    kind: "select",
    options: paymentStatusOptions,
  },
  utilisation: {
    name: "utilisation",
    label: "Usage du vehicule",
    description: "Comment le vehicule est-il utilise le plus souvent ?",
    kind: "select",
    options: usageOptions,
  },
  code_postal: {
    name: "code_postal",
    label: "Code postal",
    description: "Le code postal principal de stationnement.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "75015",
  },
  conducteur2: {
    name: "conducteur2",
    label: "Deuxieme conducteur",
    description: "Precisez si un autre conducteur utilisera regulierement le vehicule.",
    kind: "select",
    options: secondDriverOptions,
  },
  age_conducteur1: {
    name: "age_conducteur1",
    label: "Age du conducteur principal",
    description: "Renseignez l'age du conducteur principal.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "66",
  },
  age_conducteur2: {
    name: "age_conducteur2",
    label: "Age du deuxieme conducteur",
    description: "Renseignez l'age du second conducteur.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "38",
  },
  sex_conducteur1: {
    name: "sex_conducteur1",
    label: "Sexe du conducteur principal",
    description: "Selectionnez le profil correspondant.",
    kind: "select",
    options: genderOptions,
  },
  sex_conducteur2: {
    name: "sex_conducteur2",
    label: "Sexe du deuxieme conducteur",
    description: "Selectionnez le profil correspondant.",
    kind: "select",
    options: genderOptions,
  },
  anciennete_permis1: {
    name: "anciennete_permis1",
    label: "Anciennete du permis principal",
    description: "Nombre d'annees depuis l'obtention du permis.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "34",
  },
  anciennete_permis2: {
    name: "anciennete_permis2",
    label: "Anciennete du deuxieme permis",
    description: "Nombre d'annees depuis l'obtention du second permis.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "12",
  },
  anciennete_vehicule: {
    name: "anciennete_vehicule",
    label: "Age du vehicule",
    description: "Renseignez l'age du vehicule en annees.",
    kind: "input",
    inputMode: "decimal",
    placeholder: "16",
  },
  cylindre_vehicule: {
    name: "cylindre_vehicule",
    label: "Cylindree",
    description: "La cylindree du vehicule en cm3.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "1239",
  },
  din_vehicule: {
    name: "din_vehicule",
    label: "Puissance DIN",
    description: "La puissance moteur en chevaux DIN.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "55",
  },
  essence_vehicule: {
    name: "essence_vehicule",
    label: "Motorisation",
    description: "Choisissez la motorisation principale du vehicule.",
    kind: "select",
    options: fuelOptions,
  },
  marque_vehicule: {
    name: "marque_vehicule",
    label: "Marque",
    description: "Selectionnez la marque du vehicule.",
    kind: "select",
    options: brandOptions,
  },
  modele_vehicule: {
    name: "modele_vehicule",
    label: "Modele",
    description: "Choisissez ensuite le modele correspondant.",
    kind: "select",
  },
  debut_vente_vehicule: {
    name: "debut_vente_vehicule",
    label: "Debut de commercialisation",
    description: "Annee de debut de commercialisation du modele.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "16",
  },
  fin_vente_vehicule: {
    name: "fin_vente_vehicule",
    label: "Fin de commercialisation",
    description: "Annee de fin de commercialisation du modele.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "15",
  },
  vitesse_vehicule: {
    name: "vitesse_vehicule",
    label: "Vitesse maximale",
    description: "La vitesse maximale annoncee pour le vehicule.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "150",
  },
  type_vehicule: {
    name: "type_vehicule",
    label: "Categorie du vehicule",
    description: "Precisez la categorie principale du vehicule.",
    kind: "select",
    options: vehicleTypeOptions,
  },
  prix_vehicule: {
    name: "prix_vehicule",
    label: "Valeur du vehicule",
    description: "La valeur de reference du vehicule en euros.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "10321",
  },
  poids_vehicule: {
    name: "poids_vehicule",
    label: "Poids",
    description: "Le poids du vehicule en kilogrammes.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "830",
  },
};

export const quoteSteps: QuoteStep[] = [
  {
    id: "profile",
    eyebrow: "Etape 1",
    title: "Profil et formule",
    description: "Renseignez les informations principales du contrat et la formule souhaitee.",
    fields: [
      "type_contrat",
      "bonus",
      "freq_paiement",
      "paiement",
      "utilisation",
      "code_postal",
      "duree_contrat",
      "anciennete_info",
    ],
  },
  {
    id: "drivers",
    eyebrow: "Etape 2",
    title: "Conducteurs",
    description: "Decrivez le conducteur principal et, si besoin, le second conducteur.",
    fields: [
      "conducteur2",
      "age_conducteur1",
      "sex_conducteur1",
      "anciennete_permis1",
      "age_conducteur2",
      "sex_conducteur2",
      "anciennete_permis2",
    ],
  },
  {
    id: "vehicle",
    eyebrow: "Etape 3",
    title: "Vehicule",
    description: "Terminez avec les caracteristiques du vehicule pour obtenir votre estimation.",
    fields: [
      "marque_vehicule",
      "modele_vehicule",
      "anciennete_vehicule",
      "essence_vehicule",
      "type_vehicule",
      "cylindre_vehicule",
      "din_vehicule",
      "debut_vente_vehicule",
      "fin_vente_vehicule",
      "vitesse_vehicule",
      "prix_vehicule",
      "poids_vehicule",
    ],
  },
];

export function isSecondDriverEnabled(values: Pick<QuoteFormValues, "conducteur2">) {
  return values.conducteur2 === "Yes";
}

export function getStepFields(stepId: QuoteStepId, values: QuoteFormValues) {
  const fields = quoteSteps.find((step) => step.id === stepId)?.fields ?? [];

  if (stepId !== "drivers" || isSecondDriverEnabled(values)) {
    return fields;
  }

  return fields.filter(
    (fieldName) =>
      fieldName !== "age_conducteur2" &&
      fieldName !== "sex_conducteur2" &&
      fieldName !== "anciennete_permis2",
  );
}

export function getFieldOptions(fieldName: QuoteFieldName, values: QuoteFormValues) {
  if (fieldName === "modele_vehicule") {
    return getModelOptions(values.marque_vehicule);
  }

  return quoteFieldConfigs[fieldName].options ?? [];
}

function parseInteger(value: string) {
  return Number.parseInt(value, 10);
}

export function toPredictionInput(values: QuoteFormValues): PredictionInput {
  const hasSecondDriver = isSecondDriverEnabled(values);

  return {
    bonus: Number(values.bonus),
    type_contrat: values.type_contrat.trim(),
    duree_contrat: parseInteger(values.duree_contrat),
    anciennete_info: parseInteger(values.anciennete_info),
    freq_paiement: values.freq_paiement.trim(),
    paiement: values.paiement.trim(),
    utilisation: values.utilisation.trim(),
    code_postal: values.code_postal.trim(),
    conducteur2: values.conducteur2.trim(),
    age_conducteur1: parseInteger(values.age_conducteur1),
    age_conducteur2: hasSecondDriver ? parseInteger(values.age_conducteur2) : 0,
    sex_conducteur1: values.sex_conducteur1.trim(),
    sex_conducteur2: hasSecondDriver ? values.sex_conducteur2.trim() : "",
    anciennete_permis1: parseInteger(values.anciennete_permis1),
    anciennete_permis2: hasSecondDriver ? parseInteger(values.anciennete_permis2) : 0,
    anciennete_vehicule: Number(values.anciennete_vehicule),
    cylindre_vehicule: parseInteger(values.cylindre_vehicule),
    din_vehicule: parseInteger(values.din_vehicule),
    essence_vehicule: values.essence_vehicule.trim(),
    marque_vehicule: values.marque_vehicule.trim(),
    modele_vehicule: values.modele_vehicule.trim(),
    debut_vente_vehicule: parseInteger(values.debut_vente_vehicule),
    fin_vente_vehicule: parseInteger(values.fin_vente_vehicule),
    vitesse_vehicule: parseInteger(values.vitesse_vehicule),
    type_vehicule: values.type_vehicule.trim(),
    prix_vehicule: parseInteger(values.prix_vehicule),
    poids_vehicule: parseInteger(values.poids_vehicule),
  };
}

export function toFormValues(payload: PredictionInput): QuoteFormValues {
  const hasSecondDriver = payload.conducteur2 === "Yes";

  return {
    bonus: String(payload.bonus),
    type_contrat: payload.type_contrat,
    duree_contrat: String(payload.duree_contrat),
    anciennete_info: String(payload.anciennete_info),
    freq_paiement: payload.freq_paiement,
    paiement: payload.paiement,
    utilisation: payload.utilisation,
    code_postal: String(payload.code_postal),
    conducteur2: payload.conducteur2,
    age_conducteur1: String(payload.age_conducteur1),
    age_conducteur2: hasSecondDriver ? String(payload.age_conducteur2) : "",
    sex_conducteur1: payload.sex_conducteur1,
    sex_conducteur2: hasSecondDriver ? payload.sex_conducteur2 : "",
    anciennete_permis1: String(payload.anciennete_permis1),
    anciennete_permis2: hasSecondDriver ? String(payload.anciennete_permis2) : "",
    anciennete_vehicule: String(payload.anciennete_vehicule),
    cylindre_vehicule: String(payload.cylindre_vehicule),
    din_vehicule: String(payload.din_vehicule),
    essence_vehicule: payload.essence_vehicule,
    marque_vehicule: payload.marque_vehicule,
    modele_vehicule: payload.modele_vehicule,
    debut_vente_vehicule: String(payload.debut_vente_vehicule),
    fin_vente_vehicule: String(payload.fin_vente_vehicule),
    vitesse_vehicule: String(payload.vitesse_vehicule),
    type_vehicule: payload.type_vehicule,
    prix_vehicule: String(payload.prix_vehicule),
    poids_vehicule: String(payload.poids_vehicule),
  };
}

export function formatQuoteFieldValue(fieldName: QuoteFieldName, value: string | number | null | undefined) {
  if (value === null || value === undefined || value === "") {
    return "Non renseigne";
  }

  switch (fieldName) {
    case "type_contrat":
      return getChoiceLabel("type_contrat", String(value));
    case "freq_paiement":
      return getChoiceLabel("freq_paiement", String(value));
    case "paiement":
      return getChoiceLabel("paiement", String(value));
    case "utilisation":
      return getChoiceLabel("utilisation", String(value));
    case "conducteur2":
      return getChoiceLabel("conducteur2", String(value));
    case "sex_conducteur1":
    case "sex_conducteur2":
      return getChoiceLabel(fieldName, String(value));
    case "essence_vehicule":
      return getChoiceLabel("essence_vehicule", String(value));
    case "type_vehicule":
      return getChoiceLabel("type_vehicule", String(value));
    case "marque_vehicule":
    case "modele_vehicule":
      return formatTitleCase(String(value));
    case "prix_vehicule":
      return formatCurrency(Number(value));
    case "bonus":
      return String(value);
    case "code_postal":
      return String(value);
    case "duree_contrat":
    case "anciennete_info":
    case "age_conducteur1":
    case "age_conducteur2":
    case "anciennete_permis1":
    case "anciennete_permis2":
    case "debut_vente_vehicule":
    case "fin_vente_vehicule":
    case "vitesse_vehicule":
    case "poids_vehicule":
    case "cylindre_vehicule":
    case "din_vehicule":
      return formatInteger(Number(value));
    case "anciennete_vehicule":
      return String(value);
    default:
      return String(value);
  }
}
