/**
 * Validateur avancé d'objets ETF
 * @module etfValidator
 * @param {Object} etf - L'objet ETF à valider
 * @param {Object} [options] - Options de configuration
 * @param {Array} [options.requiredFields=['name', 'isin', 'category']] - Champs obligatoires
 * @param {Array} [options.numericFields=['fees', 'performance']] - Champs numériques
 * @param {Object} [options.fieldValidators={}] - Validateurs personnalisés par champ
 * @returns {Object} - { isValid: boolean, errors: Array<string> }
 */
export const validateETF = (etf, options = {}) => {
  // Configuration par défaut avec merge des options
  const config = {
    requiredFields: ['name', 'isin', 'category'],
    numericFields: ['fees', 'performance'],
    fieldValidators: {},
    ...options
  };

  const errors = [];

  // Validation de la structure de base
  if (!etf || typeof etf !== 'object' || Array.isArray(etf)) {
    return {
      isValid: false,
      errors: ['Invalid ETF object structure']
    };
  }

  // Validation des champs obligatoires
  config.requiredFields.forEach(field => {
    if (!(field in etf) || etf[field] === '' || etf[field] === null) {
      errors.push(`Missing required field: ${field}`);
    }
  });

  // Validation des champs numériques
  config.numericFields.forEach(field => {
    if (field in etf && etf[field] !== undefined) {
      if (typeof etf[field] !== 'number' || isNaN(etf[field])) {
        errors.push(`Invalid numeric value for field: ${field}`);
      }
    }
  });

  // Validation personnalisée par champ
  Object.entries(config.fieldValidators).forEach(([field, validator]) => {
    if (field in etf && !validator(etf[field])) {
      errors.push(`Validation failed for field: ${field}`);
    }
  });

  // Validation ISIN standard (exemple de validateur spécifique)
  if (etf.isin && !/^[A-Z]{2}[A-Z0-9]{9}[0-9]$/.test(etf.isin)) {
    errors.push('Invalid ISIN format');
  }

  return {
    isValid: errors.length === 0,
    errors: errors.length ? errors : null
  };
};

/**
 * Validateur réutilisable pour les champs numériques avec contraintes
 * @param {number} value - Valeur à valider
 * @param {Object} constraints - Contraintes min/max
 * @returns {boolean}
 */
export const validateNumericField = (value, constraints = {}) => {
  if (typeof value !== 'number' || isNaN(value)) return false;
  if (constraints.min !== undefined && value < constraints.min) return false;
  if (constraints.max !== undefined && value > constraints.max) return false;
  return true;
};

// Exemple d'utilisation :
/*
const etf = {
  name: 'Tech ETF',
  isin: 'FR0010478248',
  category: 'Technology',
  fees: 0.15,
  performance: 12.5
};

const result = validateETF(etf, {
  numericFields: ['fees', 'performance', 'volume'],
  fieldValidators: {
    fees: value => validateNumericField(value, { max: 2 }),
    performance: value => validateNumericField(value, { min: -100 })
  }
});

console.log(result);
*/