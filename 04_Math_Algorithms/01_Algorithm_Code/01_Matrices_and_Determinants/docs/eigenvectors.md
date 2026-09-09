## 1. The Anatomy of Power Iteration: Post-Mortem of the Custom Engine

When building a custom Math Engine for extracting Eigenvectors (instead of relying on `NumPy`), three major architectural questions arise. Here is the breakdown of what actually happens under the hood.

### 1. Why do we only get ONE Eigenvector? (Custom Engine vs. NumPy)
A $10 \times 10$ matrix mathematically contains 10 distinct Eigenvectors. However, our Power Iteration algorithm only extracts exactly one. Why?

*   **The Dominant Eigenvector:** Our loop-based algorithm naturally gravitates toward the **"Dominant Eigenvector"** (the arrow with the highest stretch factor/Eigenvalue). When you strike a vector 100 times, the direction with the strongest pull swallows the influence of all other directions (like a massive magnet overpowering smaller ones).
*   **How NumPy does it:** `np.linalg.eig()` uses a heavy, complex mathematical sequence called the **QR Algorithm**, which constantly factors the matrix to extract all 10 Eigenvectors simultaneously.
*   **The Hacker's Expansion (Deflation Method):** Can our custom engine find the other 9? Yes! Using a technique called **Matrix Deflation**, once we find the dominant eigenvector, we mathematically "subtract" its influence from the original matrix. Running the Power Iteration loop again on this new matrix will yield the *second* largest eigenvector, and so on.

---

### 2. The Math Magic: How does `max_val` become the Eigenvalue?
Using `max_val` (Infinity Norm) to normalize the vector isn't just a hack to prevent memory overflow; it is pure mathematical elegance. Visualize the very end of the loop (e.g., the 99th to 100th iteration):

**Step 1: The 99th Loop (Settled State)**
By the 99th iteration, the arrow has aligned with the true Eigenvector direction. You normalize it, forcing its largest component to be exactly `1.0`.
Let's assume our normalized vector is:
$$ v_{old} = \begin{bmatrix} -0.5 \\ 1.0 \end{bmatrix} $$

**Step 2: The 100th Loop (Matrix Action)**
You multiply the matrix $A$ by this settled vector. Since it is an Eigenvector, the matrix will NOT rotate it; it will only stretch it by the Eigenvalue ($\lambda$).
$$ A \times v_{old} = \lambda \times \begin{bmatrix} -0.5 \\ 1.0 \end{bmatrix} $$

**Step 3: The Resulting Vector**
After multiplication, the new vector looks like this:
$$ v_{new} = \begin{bmatrix} -0.5\lambda \\ 1.0\lambda \end{bmatrix} $$

**Step 4: The `max_val` Extraction**
Our code runs: `max_val = max(abs(v_new))`. This function scans the vector for the largest absolute number.
Because we forced the largest number to be `1.0` in the previous step, the largest number now is exactly **$1.0\lambda$ (which is simply $\lambda$)**!
*Conclusion: The scaling factor we use to prevent crashes is literally the exact Eigenvalue.*

---

### 3. The Physical Relationship: Matrix and the Eigenvector
To truly understand this relationship, imagine a **Tornado (बवंडर)**.

*   **The Matrix:** Think of your $10 \times 10$ matrix as a 10-Dimensional tornado. It has its own gravity, wind currents, and geometric pull.
*   **The Dummy Vector:** The initial random vector `np.random.random()` is like throwing a dry leaf into this tornado.
*   **The Loops (Multiplication):** With every loop (action), the leaf is violently tossed and pulled by the tornado's wind currents.
*   **The Dominant Eigenvector:** After 100 loops, the leaf stabilizes. It gets caught in the **Core Current** (the strongest, fastest wind tunnel of the tornado). This final direction is the Dominant Eigenvector. It represents the "soul" or the most powerful directional force of that entire matrix space.

**Application in Artificial Intelligence:**
This exact logic is the foundation of **PCA (Principal Component Analysis)** in Machine Learning. When dealing with massive datasets (like images or human genetics), AI uses this method to find the "Dominant Arrow." That arrow represents the most critical pattern in the data (e.g., the general shape of a face in an image dataset), allowing the AI to compress data without losing its soul.