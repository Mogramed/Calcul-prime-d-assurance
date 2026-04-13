import { z } from "zod";

import type { PredictionInput } from "@/generated/client/types.gen";
import {
  brandOptions,
  contractOptions,
  findVehicleVariantId,
  genderOptions,
  getChoiceLabel,
  getModelOptions,
  getVehicleVariant,
  getVehicleVariantLabel,
  getVehicleVariantOptions,
  paymentFrequencyOptions,
  paymentStatusOptions,
  secondDriverOptions,
  type QuoteOption,
  usageOptions,
  vehicleTypeOptions,
} from "@/lib/quote-catalog";
import { formatCurrency, formatInteger, formatTitleCase } from "@/lib/format";

const LEGAL_DRIVING_AGE = 18;
const MAX_DRIVER_AGE = 100;

const requiredText = z.string().trim().min(1, "Champ requis");
const requiredInteger = z
  .string()
  .trim()
  .min(1, "Champ requis")
  .refine((value) => /^\d+$/.test(value), "Entier requis");
const positiveInteger = requiredInteger.refine((value) => Number.parseInt(value, 10) > 0, "Valeur invalide");
const requiredNumber = z
  .string()
  .trim()
  .min(1, "Champ requis")
  .refine((value) => /^\d+(\.\d+)?$/.test(value), "Nombre requis");
const positiveNumber = requiredNumber.refine((value) => Number(value) > 0, "Valeur invalide");
const nonNegativeNumber = requiredNumber.refine((value) => Number(value) >= 0, "Valeur invalide");
const optionalInteger = z
  .string()
  .trim()
  .refine((value) => value === "" || /^\d+$/.test(value), "Entier invalide");

export const quoteFormSchema = z
  .object({
    bonus: requiredNumber.refine((value) => {
      const numericValue = Number(value);
      return numericValue >= 0.5 && numericValue <= 3.5;
    }, "Le bonus-malus doit etre compris entre 0,50 et 3,50."),
    type_contrat: requiredText,
    duree_contrat: positiveInteger.refine(
      (value) => Number.parseInt(value, 10) <= 50,
      "La duree du contrat doit rester inferieure ou egale a 50 ans.",
    ),
    anciennete_info: positiveInteger.refine(
      (value) => Number.parseInt(value, 10) <= 50,
      "L'anciennete du dossier doit rester inferieure ou egale a 50 ans.",
    ),
    freq_paiement: requiredText,
    paiement: requiredText,
    utilisation: requiredText,
    code_postal: z
      .string()
      .trim()
      .min(1, "Champ requis")
      .refine((value) => /^\d{4,5}$/.test(value), "Code postal invalide"),
    conducteur2: requiredText,
    age_conducteur1: requiredInteger.refine((value) => {
      const numericValue = Number.parseInt(value, 10);
      return numericValue >= LEGAL_DRIVING_AGE && numericValue <= MAX_DRIVER_AGE;
    }, "L'age du conducteur doit etre compris entre 18 et 100 ans."),
    age_conducteur2: optionalInteger,
    sex_conducteur1: requiredText,
    sex_conducteur2: z.string().trim(),
    anciennete_permis1: requiredInteger.refine((value) => {
      const numericValue = Number.parseInt(value, 10);
      return numericValue >= 0 && numericValue <= MAX_DRIVER_AGE - LEGAL_DRIVING_AGE;
    }, "L'anciennete du permis doit etre comprise entre 0 et 82 ans."),
    anciennete_permis2: optionalInteger,
    anciennete_vehicule: nonNegativeNumber.refine(
      (value) => Number(value) <= 80,
      "L'age du vehicule doit rester inferieur ou egal a 80 ans.",
    ),
    marque_vehicule: requiredText,
    modele_vehicule: requiredText,
    vehicle_variant_id: requiredText,
    cylindre_vehicule: requiredInteger,
    din_vehicule: requiredInteger,
    essence_vehicule: requiredText,
    debut_vente_vehicule: requiredInteger.refine(
      (value) => Number.parseInt(value, 10) <= 99,
      "La valeur doit rester comprise entre 0 et 99.",
    ),
    fin_vente_vehicule: requiredInteger.refine(
      (value) => Number.parseInt(value, 10) <= 99,
      "La valeur doit rester comprise entre 0 et 99.",
    ),
    vitesse_vehicule: requiredInteger,
    type_vehicule: requiredText,
    prix_vehicule: positiveInteger.refine(
      (value) => Number.parseInt(value, 10) <= 500_000,
      "La valeur du vehicule doit rester inferieure ou egale a 500 000 EUR.",
    ),
    poids_vehicule: requiredInteger,
  })
  .superRefine((values, context) => {
    const ageConducteur1 = Number.parseInt(values.age_conducteur1, 10);
    const anciennetePermis1 = Number.parseInt(values.anciennete_permis1, 10);

    if (!Number.isNaN(ageConducteur1) && !Number.isNaN(anciennetePermis1)) {
      const maxAnciennetePermis1 = ageConducteur1 - LEGAL_DRIVING_AGE;
      if (anciennetePermis1 > maxAnciennetePermis1) {
        context.addIssue({
          code: "custom",
          path: ["anciennete_permis1"],
          message: `Avec un conducteur de ${ageConducteur1} ans, l'anciennete maximale du permis est de ${Math.max(maxAnciennetePermis1, 0)} ans.`,
        });
      }
    }

    if (values.conducteur2 === "Yes") {
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

      if (values.age_conducteur2.trim()) {
        const ageConducteur2 = Number.parseInt(values.age_conducteur2, 10);
        if (Number.isNaN(ageConducteur2) || ageConducteur2 < LEGAL_DRIVING_AGE || ageConducteur2 > MAX_DRIVER_AGE) {
          context.addIssue({
            code: "custom",
            path: ["age_conducteur2"],
            message: "L'age du deuxieme conducteur doit etre compris entre 18 et 100 ans.",
          });
        }
      }

      if (values.age_conducteur2.trim() && values.anciennete_permis2.trim()) {
        const ageConducteur2 = Number.parseInt(values.age_conducteur2, 10);
        const anciennetePermis2 = Number.parseInt(values.anciennete_permis2, 10);
        if (!Number.isNaN(ageConducteur2) && !Number.isNaN(anciennetePermis2)) {
          const maxAnciennetePermis2 = ageConducteur2 - LEGAL_DRIVING_AGE;
          if (anciennetePermis2 > maxAnciennetePermis2) {
            context.addIssue({
              code: "custom",
              path: ["anciennete_permis2"],
              message: `Avec un conducteur de ${ageConducteur2} ans, l'anciennete maximale du permis est de ${Math.max(maxAnciennetePermis2, 0)} ans.`,
            });
          }
        }
      }
    }

    const variant = getVehicleVariant(
      values.marque_vehicule,
      values.modele_vehicule,
      values.vehicle_variant_id,
    );

    if (!variant) {
      context.addIssue({
        code: "custom",
        path: ["vehicle_variant_id"],
        message: "Choisissez une configuration valide pour ce vehicule.",
      });
      return;
    }

    const expectedSpecs = {
      cylindre_vehicule: String(variant.cylindre_vehicule),
      din_vehicule: String(variant.din_vehicule),
      essence_vehicule: variant.essence_vehicule,
      vitesse_vehicule: String(variant.vitesse_vehicule),
      poids_vehicule: String(variant.poids_vehicule),
    } satisfies Record<string, string>;

    for (const [fieldName, expectedValue] of Object.entries(expectedSpecs)) {
      if (String(values[fieldName as keyof typeof values]).trim() !== expectedValue) {
        context.addIssue({
          code: "custom",
          path: [fieldName],
          message: "Cette caracteristique doit correspondre au vehicule selectionne.",
        });
      }
    }
  });

export type QuoteFormValues = z.infer<typeof quoteFormSchema>;
export type QuoteFieldName = keyof QuoteFormValues;
export type QuoteStepId = "profile" | "drivers" | "vehicle";

type FieldKind = "input" | "select" | "derived";

export type QuoteFieldConfig = {
  name: QuoteFieldName;
  label: string;
  description: string;
  kind: FieldKind;
  placeholder?: string;
  inputMode?: "text" | "decimal" | "numeric";
  min?: number;
  max?: number;
  step?: number;
  options?: readonly QuoteOption[];
};

export type QuoteStep = {
  id: QuoteStepId;
  eyebrow: string;
  title: string;
  description: string;
  fields: QuoteFieldName[];
};

export const vehicleAutofillFieldNames = [
  "essence_vehicule",
  "din_vehicule",
  "vitesse_vehicule",
  "cylindre_vehicule",
  "poids_vehicule",
] as const satisfies readonly QuoteFieldName[];

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
  marque_vehicule: "",
  modele_vehicule: "",
  vehicle_variant_id: "",
  cylindre_vehicule: "",
  din_vehicule: "",
  essence_vehicule: "",
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
    min: 0.5,
    max: 3.5,
    step: 0.01,
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
    min: 1,
    max: 50,
    step: 1,
  },
  anciennete_info: {
    name: "anciennete_info",
    label: "Anciennete du dossier",
    description: "Nombre d'annees d'historique disponible pour ce contrat.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "1",
    min: 1,
    max: 50,
    step: 1,
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
    placeholder: "45",
    min: 18,
    max: 100,
    step: 1,
  },
  age_conducteur2: {
    name: "age_conducteur2",
    label: "Age du deuxieme conducteur",
    description: "Renseignez l'age du second conducteur.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "38",
    min: 18,
    max: 100,
    step: 1,
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
    placeholder: "20",
    min: 0,
    max: 82,
    step: 1,
  },
  anciennete_permis2: {
    name: "anciennete_permis2",
    label: "Anciennete du deuxieme permis",
    description: "Nombre d'annees depuis l'obtention du second permis.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "12",
    min: 0,
    max: 82,
    step: 1,
  },
  anciennete_vehicule: {
    name: "anciennete_vehicule",
    label: "Age du vehicule",
    description: "Renseignez l'age du vehicule en annees.",
    kind: "input",
    inputMode: "decimal",
    placeholder: "8",
    min: 0,
    max: 80,
    step: 0.1,
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
  vehicle_variant_id: {
    name: "vehicle_variant_id",
    label: "Configuration technique",
    description:
      "Choisissez la version qui correspond a votre vehicule. Les caracteristiques techniques les plus sensibles seront ensuite renseignees automatiquement.",
    kind: "select",
  },
  cylindre_vehicule: {
    name: "cylindre_vehicule",
    label: "Cylindree",
    description: "Renseignement automatique selon la version selectionnee.",
    kind: "derived",
  },
  din_vehicule: {
    name: "din_vehicule",
    label: "Puissance DIN",
    description: "Renseignement automatique selon la version selectionnee.",
    kind: "derived",
  },
  essence_vehicule: {
    name: "essence_vehicule",
    label: "Motorisation",
    description: "Renseignement automatique selon la version selectionnee.",
    kind: "derived",
  },
  debut_vente_vehicule: {
    name: "debut_vente_vehicule",
    label: "Debut de commercialisation",
    description: "Indiquez le repere de debut de serie du vehicule.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "16",
    min: 0,
    max: 99,
    step: 1,
  },
  fin_vente_vehicule: {
    name: "fin_vente_vehicule",
    label: "Fin de commercialisation",
    description: "Indiquez le repere de fin de serie correspondant.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "15",
    min: 0,
    max: 99,
    step: 1,
  },
  vitesse_vehicule: {
    name: "vitesse_vehicule",
    label: "Vitesse maximale",
    description: "Renseignement automatique selon la version selectionnee.",
    kind: "derived",
  },
  type_vehicule: {
    name: "type_vehicule",
    label: "Categorie du vehicule",
    description: "Choisissez la categorie la plus proche de votre vehicule.",
    kind: "select",
    options: vehicleTypeOptions,
  },
  prix_vehicule: {
    name: "prix_vehicule",
    label: "Valeur de reference",
    description: "Renseignez la valeur de reference du vehicule.",
    kind: "input",
    inputMode: "numeric",
    placeholder: "10321",
    min: 1,
    max: 500000,
    step: 1,
  },
  poids_vehicule: {
    name: "poids_vehicule",
    label: "Poids",
    description: "Poids historique associe a la version selectionnee.",
    kind: "derived",
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
    description:
      "Choisissez votre modele puis la configuration la plus proche. Les donnees techniques les plus sensibles sont ensuite remplies automatiquement.",
    fields: [
      "marque_vehicule",
      "modele_vehicule",
      "vehicle_variant_id",
      "anciennete_vehicule",
      "essence_vehicule",
      "din_vehicule",
      "vitesse_vehicule",
      "debut_vente_vehicule",
      "fin_vente_vehicule",
      "type_vehicule",
      "cylindre_vehicule",
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

export function clearVehicleAutofillValues(
  values: QuoteFormValues,
): QuoteFormValues {
  const nextValues = { ...values };
  for (const fieldName of vehicleAutofillFieldNames) {
    nextValues[fieldName] = "";
  }
  return nextValues;
}

export function applyVehicleVariantToValues(
  values: QuoteFormValues,
): QuoteFormValues {
  const variant = getVehicleVariant(
    values.marque_vehicule,
    values.modele_vehicule,
    values.vehicle_variant_id,
  );

  if (!variant) {
    return clearVehicleAutofillValues(values);
  }

  return {
    ...values,
    essence_vehicule: variant.essence_vehicule,
    din_vehicule: String(variant.din_vehicule),
    vitesse_vehicule: String(variant.vitesse_vehicule),
    cylindre_vehicule: String(variant.cylindre_vehicule),
    poids_vehicule: String(variant.poids_vehicule),
  };
}

export function getFieldOptions(fieldName: QuoteFieldName, values: QuoteFormValues) {
  if (fieldName === "modele_vehicule") {
    return getModelOptions(values.marque_vehicule);
  }

  if (fieldName === "vehicle_variant_id") {
    return getVehicleVariantOptions(values.marque_vehicule, values.modele_vehicule);
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
  const baseValues: QuoteFormValues = {
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
    marque_vehicule: payload.marque_vehicule,
    modele_vehicule: payload.modele_vehicule,
    vehicle_variant_id: "",
    cylindre_vehicule: String(payload.cylindre_vehicule),
    din_vehicule: String(payload.din_vehicule),
    essence_vehicule: payload.essence_vehicule,
    debut_vente_vehicule: String(payload.debut_vente_vehicule),
    fin_vente_vehicule: String(payload.fin_vente_vehicule),
    vitesse_vehicule: String(payload.vitesse_vehicule),
    type_vehicule: payload.type_vehicule,
    prix_vehicule: String(payload.prix_vehicule),
    poids_vehicule: String(payload.poids_vehicule),
  };

  return {
    ...baseValues,
    vehicle_variant_id: findVehicleVariantId(baseValues),
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
    case "vehicle_variant_id":
      return getVehicleVariantLabel(String(value)) ?? String(value);
    case "marque_vehicule":
    case "modele_vehicule":
      return formatTitleCase(String(value));
    case "prix_vehicule":
      return formatCurrency(Number(value));
    case "bonus":
    case "anciennete_vehicule":
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
    default:
      return String(value);
  }
}
