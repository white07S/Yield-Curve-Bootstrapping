import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
# Generate synthetic data
np.random.seed(42)
num_bonds = 20

maturities = np.arange(1, num_bonds + 1)
coupon_rates = np.random.uniform(low=0.02, high=0.06, size=num_bonds)
bond_prices = np.random.uniform(low=90, high=110, size=num_bonds)

# Create a pandas DataFrame
data = {
    'Maturity (Years)': maturities,
    'Coupon Rate': coupon_rates,
    'Bond Price': bond_prices
}
bonds = pd.DataFrame(data)

# Bootstrapping function
def bootstrap_yield_curve(bonds):
    zero_rates = []
    discount_factors = []

    for idx, bond in bonds.iterrows():
        maturity = bond['Maturity (Years)']
        coupon_rate = bond['Coupon Rate']
        bond_price = bond['Bond Price']
        periods = np.arange(1, maturity + 1)

        if idx == 0:
            zero_rate = -np.log(bond_price / (1 + coupon_rate)) / maturity
            zero_rates.append(zero_rate)
            discount_factor = np.exp(-zero_rate * maturity)
            discount_factors.append(discount_factor)
        else:
            cash_flows = coupon_rate * np.ones(int(maturity))

            cash_flows[-1] += 1

            def objective_function(zero_rate):
                discount_factors = np.exp(-zero_rate * periods)
                return np.sum(cash_flows * discount_factors) - bond_price

            zero_rate = scipy.optimize.newton(objective_function, zero_rates[-1])

            zero_rates.append(zero_rate)
            discount_factor = np.exp(-zero_rate * maturity)
            discount_factors.append(discount_factor)

    return np.array(zero_rates), np.array(discount_factors)

# Bootstrap zero-coupon yield curve
zero_rates, discount_factors = bootstrap_yield_curve(bonds)

# Add zero rates and discount factors to DataFrame
bonds['Zero Rate'] = zero_rates
bonds['Discount Factor'] = discount_factors

# Analysis and Visualization
plt.figure(figsize=(10, 6))
plt.plot(bonds['Maturity (Years)'].to_numpy(), bonds['Zero Rate'].to_numpy(), marker='o')
plt.title('Bootstrapped Zero-Coupon Yield Curve')
plt.xlabel('Maturity (Years)')
plt.ylabel('Zero Rate')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(bonds['Maturity (Years)'].to_numpy(), bonds['Discount Factor'].to_numpy(), marker='o')
plt.title('Bootstrapped Discount Factors')
plt.xlabel('Maturity (Years)')
plt.ylabel('Discount Factor')
plt.grid(True)
plt.show()
