import json
import numpy as np
import matplotlib.pyplot as plt

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO: first generate random numbers from the uniform distribution
    if distribution == "exponential":
        lamb = kwargs["lambda"]
        for _ in range(num_samples):
            num = np.random.uniform(0,1)
            newSample = round(-1 * np.log(num)/lamb, 4)
            samples.append(newSample)
    if distribution == "cauchy":
        x_0 = kwargs["peak_x"]
        gamma = kwargs["gamma"]
        for _ in range(num_samples):
            num = np.random.uniform(0,1)
            newSample = round(x_0 + gamma*(np.tan(np.pi*(num - 0.5))),4)
            samples.append(newSample)
    # END TODO
            
    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        
        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"
        plt.hist(samples, bins = "auto")
        plt.title(distribution.capitalize() +  " Distribution")
        plt.savefig("q1_" + distribution + ".png")
        plt.clf()
        # END TODO
