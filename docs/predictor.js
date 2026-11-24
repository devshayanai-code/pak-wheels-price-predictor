/* Runs the exported gradient boosting model in the browser (no server needed).
 *
 * It repeats the three steps of the scikit-learn pipeline:
 *   1. One-hot encode text inputs, scale numeric inputs  -> a row of numbers
 *   2. Walk every tree, adding up the leaf values          -> log(1 + price)
 *   3. Undo the log                                         -> price in rupees
 * Works in the browser and in Node (for tests).
 */
(function (root) {
  const ROUND_TO = 10000;          // round like real listings
  const MAX_SIMILAR = 3;
  const SIMILAR_YEAR_WINDOW = 2;
  const KM_PER_YEAR = 50000;       // 50,000 km apart counts like 1 year apart

  function encode(model, car) {
    const index = new Map(model.columns.map((name, i) => [name, i]));
    const row = new Float64Array(model.columns.length);
    for (const feature of model.categorical) {
      const exact = index.get(`categorical__${feature}_${car[feature]}`);
      const rare = index.get(`categorical__${feature}_infrequent_sklearn`);
      if (exact !== undefined) row[exact] = 1;
      else if (rare !== undefined) row[rare] = 1;  // rare or unseen value
    }
    model.numeric.forEach((feature, i) => {
      const scaled = (Number(car[feature]) - model.scaler_mean[i]) / model.scaler_scale[i];
      row[index.get(`numeric__${feature}`)] = scaled;
    });
    return row;
  }

  function walkTree(nodes, row) {
    let node = nodes[0];
    while (node[0] !== -1) {  // [feature, threshold, left, right, missingLeft, value]
      const x = row[node[0]];
      const goLeft = Number.isNaN(x) ? node[4] === 1 : x <= node[1];
      node = nodes[goLeft ? node[2] : node[3]];
    }
    return node[5];
  }

  function predictRaw(model, car) {
    const row = encode(model, car);
    let logPrice = model.baseline;
    for (const tree of model.trees) logPrice += walkTree(tree, row);
    return Math.expm1(logPrice);
  }

  const roundPrice = (v) => Math.round(v / ROUND_TO) * ROUND_TO;

  function similarListings(listings, car) {
    return listings
      .filter(([make, model, year]) =>
        make === car.make && model === car.model && Math.abs(year - car.year) <= SIMILAR_YEAR_WINDOW)
      .map((l) => ({ l, d: Math.abs(l[2] - car.year) + Math.abs(l[3] - car.mileage) / KM_PER_YEAR }))
      .sort((a, b) => a.d - b.d)
      .slice(0, MAX_SIMILAR)
      .map(({ l }) => ({ year: l[2], mileage: l[3], city: l[4], price: l[5] }));
  }

  function predictPrice(model, listings, car) {
    const price = predictRaw(model, car);
    const errorRate = model.metrics["test_mape_%"] / 100;
    return {
      predicted_price: roundPrice(price),
      price_range: { low: roundPrice(price * (1 - errorRate)), high: roundPrice(price * (1 + errorRate)) },
      average_error_percent: Math.round(errorRate * 1000) / 10,
      similar_listings: similarListings(listings, car),
    };
  }

  /* Same checks as the Django API's CarInputSerializer. Returns {field: message}. */
  function validate(options, car) {
    const errors = {};
    const thisYear = new Date().getFullYear();
    if (!options.makes[car.make]) errors.make = `Unknown make '${car.make}'.`;
    else if (!options.makes[car.make].includes(car.model)) errors.model = `Unknown model '${car.model}'.`;
    if (!Number.isInteger(car.year) || car.year < 1980 || car.year > thisYear + 1)
      errors.year = `Enter a year between 1980 and ${thisYear + 1}.`;
    if (!Number.isInteger(car.mileage) || car.mileage < 0 || car.mileage > 500000)
      errors.mileage = "Enter mileage between 0 and 500,000 km.";
    if (!Number.isInteger(car.engine_cc) || car.engine_cc < 0 || car.engine_cc > 7000)
      errors.engine_cc = "Enter engine size between 0 and 7,000 cc.";
    else if (car.fuel !== "Electric" && car.engine_cc === 0)
      errors.engine_cc = "Engine size is required for non-electric cars.";
    if (!options.fuels.includes(car.fuel)) errors.fuel = `Unknown fuel '${car.fuel}'.`;
    if (!options.transmissions.includes(car.transmission))
      errors.transmission = `Unknown transmission '${car.transmission}'.`;
    if (!options.cities.includes(car.city)) car.city = "Other";  // grouped as in training
    return errors;
  }

  const api = { predictRaw, predictPrice, validate };
  if (typeof module !== "undefined") module.exports = api;
  else root.Predictor = api;
})(this);
