from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

MU_U = 10
VAR_U = 1
SAMPLES = 1000
MU_M= np.array([0, 0, 4, 0])
COV_M = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(MU_U, VAR_U, SAMPLES)
    UNI = UnivariateGaussian()
    UNI = UNI.fit(X)
    print("(" + str(UNI.mu_) + ", " + str(UNI.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    sample_size = np.linspace(10, 1000, 100)
    UNI_CHANGING = UnivariateGaussian()
    estim_expect = np.zeros(100)
    for i in range(100):
        UNI_CHANGING.fit(X[:int(sample_size[i])])
        estim_expect[i] = abs(mu - UNI_CHANGING.mu_)
    plt.plot(sample_size, estim_expect)
    plt.xlabel("sample size")
    plt.ylabel("absolute distance")
    plt.title("distance between the estimated and true value of expectation\n as a function the sample size")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = UNI.pdf(X)
    plt.scatter(X, pdf, s=5)
    plt.xlabel("ordered sample value")
    plt.ylabel("PDF value")
    plt.title("empirical PDF function under the fitted model")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    X = np.random.multivariate_normal(MU_M, COV_M, SAMPLES)
    Multi = MultivariateGaussian()
    Multi.fit(X)
    print(str(Multi.mu_) + "\n" + str(Multi.cov_))

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    X_vec, Y_vec = np.meshgrid(f1, f3)
    LL = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            cur_mu = np.array([f1[i], 0, f3[j], 0])
            LL[i, j] = Multi.log_likelihood(cur_mu, sigma, X)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X_vec, Y_vec, LL)
    ax.axis([X_vec.min(), X_vec.max(), Y_vec.min(), Y_vec.max()])
    fig.colorbar(c, ax=ax)
    plt.xlabel("value of f1 (first feature expectation")
    plt.ylabel("value of f3 (third feature expectation")
    plt.show()

    # Question 6 - Maximum likelihood
    # max_ = np.argmax(LL)
    max_ind = np.unravel_index(np.argmax(LL), LL.shape)
    # max_f1 = 0
    # max_f3 = 0
    # max_val = -np.inf
    # for i in range(200):
    #     for j in range(200):
    #         if LL[i][j] >= max_val:
    #             max_f1 = f1[i]
    #             max_f3 = f3[j]
    #             max_val = LL[i][j]
    print(f1[max_ind[0]], f3[max_ind[1]])

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
