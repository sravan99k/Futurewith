# 🧠 Deep Learning Neural Networks Practice Questions & Exercises

## Simple Level Questions for Everyone!

_Based on the Deep Learning Neural Networks Simple Guide_

---

## 🎯 **SECTION A: DEEP LEARNING BASICS** (Beginner Level)

### **Question 1: Deep Learning vs Traditional ML**

Circle True or False for each statement:

1. Deep Learning uses many layers of neurons. True / False
2. Traditional ML can handle images better than Deep Learning. True / False
3. Deep Learning is inspired by how the human brain works. True / False
4. Neural networks have been around for over 100 years. True / False
5. Deep Learning can learn complex patterns automatically. True / False

**Answers:** 1-True, 2-False, 3-True, 4-True, 5-True

---

### **Question 2: Neural Network Components**

Match each component with its function:

| **Component** | **Function**                      |
| ------------- | --------------------------------- |
| Neuron        | Makes decisions based on inputs   |
| Weight        | Determines input importance       |
| Activation    | Decision rule (when to fire)      |
| Layer         | Group of neurons working together |
| Bias          | Default tendency or offset        |

---

### **Question 3: Fill in the Blanks**

Complete these sentences using the words: **layers, patterns, images, brain, complex**

1. Deep Learning networks have many ******\_****** of neurons.
2. Neural networks are inspired by the human ******\_******.
3. CNNs are especially good at processing ******\_******.
4. Deep Learning can learn very ******\_****** relationships in data.
5. Networks find ******\_****** by training on lots of examples.

**Answers:** 1-layers, 2-brain, 3-images, 4-complex, 5-patterns

---

### **Question 4: Activation Functions**

Match each activation function with its description:

| **Function** | **Description**                                |
| ------------ | ---------------------------------------------- |
| Sigmoid      | Keeps only positive values, zero for negatives |
| ReLU         | Gives probability between 0 and 1              |
| Step         | Simple yes/no decision (0 or 1)                |

**Answers:** ReLU → Keeps only positive values, Step → Simple yes/no, Sigmoid → Probability

---

## 🎯 **SECTION B: FEEDFORWARD NETWORKS** (Medium Level)

### **Question 5: MLP Architecture**

A Multi-Layer Perceptron has:

- Input layer: 4 neurons
- Hidden layer 1: 8 neurons
- Hidden layer 2: 6 neurons
- Output layer: 3 neurons

Calculate the total number of weights:

- From input to hidden1: **\_\_\_** weights
- From hidden1 to hidden2: **\_\_\_** weights
- From hidden2 to output: **\_\_\_** weights
- Total weights: **\_\_\_** weights

**Answers:** 4×8=32, 8×6=48, 6×3=18, Total=98 weights

---

### **Question 6: Forward Pass Calculation**

A simple network processes this data:

```
Input: [2, 3]
Weights: [[0.5, 0.2], [0.3, 0.8]]
Bias: [1, 0]
Activation: ReLU
```

What happens in the forward pass?

1. Weighted sum: **\_\_\_**
2. Add bias: **\_\_\_**
3. Apply ReLU: **\_\_\_**

**Answers:**

1. [2×0.5+3×0.2=1.6, 2×0.3+3×0.8=3.0]
2. [1.6+1=2.6, 3.0+0=3.0]
3. [max(0,2.6)=2.6, max(0,3.0)=3.0]

---

### **Question 7: Training Process**

Put these training steps in the correct order:

□ Calculate error/loss
□ Backward pass (adjust weights)
□ Forward pass (make prediction)
□ Update model parameters
□ Show training data
□ Repeat for many epochs

**Correct Order:** 5 → 3 → 1 → 2 → 4 → 6

---

## 🎯 **SECTION C: CONVOLUTIONAL NEURAL NETWORKS** (Medium Level)

### **Question 8: CNN vs Traditional NN**

Why are CNNs better for image processing than traditional neural networks?

1. ***
2. ***
3. ***

**Sample Answers:**

1. Can detect local patterns (edges, shapes)
2. Parameter sharing reduces memory usage
3. Translation invariance (works regardless of object position)

---

### **Question 9: Convolution Operation**

A 3×3 image convolved with a 2×2 filter:

```
Image:    Filter:    Result:
[1,2,3]   [0,1]      [?,?]
[4,5,6]   [1,0]      [?,?]
[7,8,9]
```

Calculate the convolution result:

- Position (0,0): (1×0 + 2×1 + 4×1 + 5×0) = **\_\_\_**
- Position (1,1): (5×0 + 6×1 + 8×1 + 9×0) = **\_\_\_**

**Answers:** 1×0+2×1+4×1+5×0=6, 5×0+6×1+8×1+9×0=14

---

### **Question 10: Famous CNN Architectures**

Match each CNN architecture with its main innovation:

| **Architecture** | **Innovation**                              |
| ---------------- | ------------------------------------------- |
| LeNet-5          | Residual connections for very deep networks |
| ResNet           | First successful CNN for handwritten digits |
| YOLO             | Real-time object detection in one pass      |

**Answers:** LeNet-5 → First successful CNN, ResNet → Residual connections, YOLO → Real-time detection

---

### **Question 11: CNN Applications**

For each application, choose the best CNN architecture:

1. **Medical X-ray diagnosis**
   Answer: ******\_******

2. **Real-time video surveillance**
   Answer: ******\_******

3. **Handwritten digit recognition**
   Answer: ******\_******

**Answers:** 1-ResNet, 2-YOLO, 3-LeNet-5

---

## 🎯 **SECTION D: RECURRENT NEURAL NETWORKS** (Advanced Level)

### **Question 12: RNN vs Feedforward**

Fill in the differences:

**Feedforward Networks:**

- Process each input ******\_******
- No ******\_****** between steps
- Good for ******\_****** problems

**Recurrent Networks:**

- Process inputs ******\_******
- Have ******\_****** to remember previous steps
- Good for ******\_****** problems

**Answers:** Feedforward: independently, memory, static; RNN: sequentially, memory, sequential

---

### **Question 13: Memory Problem**

Why do simple RNNs have trouble with long sequences?

1. ***
2. ***
3. ***

**Sample Answers:**

1. Vanishing gradient problem
2. Short-term memory limitations
3. Difficulty learning long-range dependencies

---

### **Question 14: Sequence Prediction**

Given this RNN processing sequence ["The", "cat", "is", "sleeping"]:

Step 1: Process "The" → Hidden state H1
Step 2: Process "cat" + H1 → Hidden state H2  
Step 3: Process "is" + H2 → Hidden state H3
Step 4: Process "sleeping" + H3 → Hidden state H4

What information does H4 contain?
Answer: ************\_************

**Answer:** Understanding of the full sentence "The cat is sleeping" (context from all previous words)

---

## 🎯 **SECTION E: LSTM NETWORKS** (Advanced Level)

### **Question 15: LSTM Gates**

Match each LSTM gate with its function:

| **Gate**    | **Function**                     |
| ----------- | -------------------------------- |
| Forget Gate | What new information to remember |
| Input Gate  | What information to forget       |
| Output Gate | What to focus on for output      |

**Answers:** Forget → What to forget, Input → What to remember, Output → What to focus on

---

### **Question 16: LSTM vs Simple RNN**

Why is LSTM better than simple RNN for long sequences?

1. **Memory Management:** ******\_******
2. **Information Flow:** ******\_******
3. **Learning Capability:** ******\_******

**Sample Answers:**

1. Can selectively forget or remember information
2. Information flows through cell state (highway)
3. Better at learning long-term dependencies

---

### **Question 17: LSTM Applications**

For each scenario, explain why LSTM is a good choice:

1. **Language Translation:**
   Answer: ************\_************

2. **Stock Price Prediction:**
   Answer: ************\_************

3. **Speech Recognition:**
   Answer: ************\_************

**Sample Answers:**

1. Needs to remember context from entire sentence
2. Must remember patterns over long time periods
3. Audio features have temporal dependencies

---

## 🎯 **SECTION F: ATTENTION MECHANISMS** (Expert Level)

### **Question 18: Attention Concept**

Explain attention in simple terms:

Attention is like ******\_****** that helps AI ******\_****** on the most important parts of ******\_****** while processing.

**Answer:** spotlight/focus, focus/attend, input/data

---

### **Question 19: Attention Components**

Match the attention components with their questions:

| **Component** | **Question**                   |
| ------------- | ------------------------------ |
| Query         | What do these contain?         |
| Key           | What information is available? |
| Value         | What am I looking for?         |

**Answers:** Query → What am I looking for, Key → What info available, Value → What do these contain

---

### **Question 20: Self-Attention Example**

In the sentence "The cat sat on the mat because it was tired":

What does "it" refer to?

Without attention: ******\_******
With attention:

- "it" → "cat" (90% attention)
- "it" → "mat" (10% attention)
  Answer: ******\_******

**Answers:** Confused/uncertain, "cat"

---

## 🎯 **SECTION G: TRANSFORMERS** (Expert Level)

### **Question 21: Transformer vs RNN**

Complete the comparison:

| **Aspect**  | **RNN**    | **Transformer** |
| ----------- | ---------- | --------------- |
| Processing  | Sequential | ******\_******  |
| Memory      | Limited    | ******\_******  |
| Speed       | Slow       | ******\_******  |
| Parallelism | Limited    | ******\_******  |

**Answers:** Parallel, Unlimited/good, Fast, Full

---

### **Question 22: Transformer Architecture**

Fill in the Transformer components:

```
Input → Embeddings →
[__________] →
[__________] →
[__________] →
Output
```

**Answers:** Position encoding, Multi-head attention, Feed-forward network

---

### **Question 23: Famous Transformers**

Match each model with its purpose:

| **Model** | **Main Use**                     |
| --------- | -------------------------------- |
| BERT      | Understanding language context   |
| GPT       | Generating human-like text       |
| T5        | All NLP tasks as text generation |

---

### **Question 24: Transformer Applications**

For each application, choose the best Transformer:

1. **Question Answering System:**
   Answer: ******\_******

2. **Creative Writing Assistant:**
   Answer: ******\_******

3. **Text Summarization:**
   Answer: ******\_******

**Answers:** 1-BERT, 2-GPT, 3-T5

---

## 🎯 **SECTION H: VISION TRANSFORMERS** (Expert Level)

### **Question 25: ViT vs CNN**

Complete the comparison:

| **Feature**                   | **CNN**        | **ViT**        |
| ----------------------------- | -------------- | -------------- |
| Data Processing               | Grid-based     | ******\_****** |
| Local Patterns                | Inductive bias | ******\_****** |
| Global Context                | Limited        | ******\_****** |
| Performance on Large Datasets | Good           | ******\_****** |

**Answers:** Patch-based, Not built-in, Excellent, Better

---

### **Question 26: Image Patching**

A 224×224 image with 16×16 patches creates how many patches?

Calculation: **\_\_\_** patches

**Answer:** (224/16) × (224/16) = 14 × 14 = 196 patches

---

### **Question 27: ViT Advantages**

Why might ViT outperform CNNs on very large datasets?

1. ***
2. ***
3. ***

**Sample Answers:**

1. Global attention mechanism
2. No inductive biases (more flexible)
3. Scales better with data and model size

---

## 🎯 **SECTION I: ADVANCED ARCHITECTURES** (Expert Level)

### **Question 28: GAN Components**

Fill in the GAN process:

```
Generator creates _____________ →
Discriminator evaluates _____________ →
Both networks _____________ each other →
Eventually Generator creates realistic _____________
```

**Answers:** fake data, real/fake, compete/improve, data/images

---

### **Question 29: Autoencoder Purpose**

What does an autoencoder learn?

1. **Input → ******\_****** → Output**
2. The network learns to ******\_****** the input
3. The compressed representation is called ******\_******

**Answers:** compressed representation, reconstruct/recreate, bottleneck/encoding

---

### **Question 30: U-Net Application**

Why is U-Net particularly good for medical image segmentation?

1. ***
2. ***
3. ***

**Sample Answers:**

1. Skip connections preserve fine details
2. Encoder-decoder structure captures both local and global features
3. Works well with limited medical training data

---

## 🎯 **SECTION J: REAL-WORLD SCENARIOS** (Advanced Level)

### **Question 31: Architecture Selection**

Choose the best architecture for each problem:

| **Problem**                       | **Best Architecture** |
| --------------------------------- | --------------------- |
| Recognize handwritten digits      | ******\_******        |
| Generate poetry                   | ******\_******        |
| Translate Spanish to English      | ******\_******        |
| Detect objects in real-time video | ******\_******        |
| Compress images for storage       | ******\_******        |

**Answers:** CNN/LeNet, RNN/LSTM/GPT, Encoder-Decoder LSTM/Transformer, YOLO, Autoencoder

---

### **Question 32: Problem Analysis**

Analyze these problems and recommend solutions:

**Problem 1: E-commerce product image classification (1000 categories)**

- Data type: ******\_******
- Challenge: ******\_******
- Recommended approach: ******\_******

**Problem 2: Chatbot for customer service**

- Data type: ******\_******
- Challenge: ******\_******
- Recommended approach: ******\_******

**Problem 3: Fraud detection in financial transactions**

- Data type: ******\_******
- Challenge: ******\_******
- Recommended approach: ******\_******

**Sample Answers:**

1. Images, many classes, CNN with transfer learning
2. Text sequences, understanding context, Transformer/RNN
3. Sequential numerical data, anomaly detection, LSTM/Autoencoder

---

### **Question 33: Model Evaluation**

You trained three models with these results:

| Model | Training Accuracy | Testing Accuracy | Speed  |
| ----- | ----------------- | ---------------- | ------ |
| A     | 99%               | 70%              | Fast   |
| B     | 85%               | 83%              | Medium |
| C     | 90%               | 89%              | Slow   |

1. **Which model is overfitting?** **\_\_\_**
2. **Which model has best generalization?** **\_\_\_**
3. **Which model would you choose and why?** **\_\_\_**

**Answers:** 1-Model A, 2-Model C, 3-Model C (best balance of accuracy and generalization)

---

### **Question 34: Resource Planning**

Your deep learning project needs to process 1 million images daily:

1. **Hardware consideration:** ******\_******
2. **Software consideration:** ******\_******
3. **Optimization strategy:** ******\_******

**Sample Answers:** GPU cluster, batch processing, model compression/distributed inference

---

## 🎯 **SECTION K: CODING CONCEPTS** (Advanced Level)

### **Question 35: PyTorch Basics**

Look at this code and answer questions:

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = SimpleNet()
input_data = torch.randn(32, 10)  # Batch of 32, 10 features
output = model(input_data)
```

1. **What does the forward method do?** **\_\_\_**
2. **What is the output shape?** **\_\_\_**
3. **What activation function is used?** **\_\_\_**

**Answers:** Defines data flow through network, (32, 2), ReLU

---

### **Question 36: Training Loop**

Complete this training loop:

```python
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # 1. Forward pass
        predictions = model(batch_x)

        # 2. Calculate loss
        loss = criterion(predictions, batch_y)

        # 3. _____________
        optimizer.zero_grad()

        # 4. _____________
        loss.backward()

        # 5. _____________
        optimizer.step()
```

**Answers:** 3-Clear gradients, 4-Backpropagate, 5-Update weights

---

### **Question 37: Model Saving**

What are the two main ways to save PyTorch models?

1. ********\_******: Saves model architecture and weights**
2. ********\_******: Saves only model weights**

**Answers:** torch.save(model, path), model.state_dict()

---

## 🎯 **SECTION L: FUN CHALLENGES** (For All Levels)

### **Challenge 1: Build Your Own Network**

Design a simple neural network for this problem:

**Problem:** Predict if a student will pass (1) or fail (0) based on:

- Study hours (0-20)
- Sleep hours (0-10)
- Attendance percentage (0-100)

**Your network design:**
Input layer: **\_\_\_** neurons
Hidden layer(s): **\_\_\_** neurons
Output layer: **\_\_\_** neurons
Activation functions: **\_\_\_**

**Sample Answer:** Input: 3, Hidden: 6, Output: 1, Activation: ReLU for hidden, Sigmoid for output

---

### **Challenge 2: Debug the Network**

This network has problems. Can you spot them?

```python
class BadCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)  # Missing batch size dimension
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64, 10)  # Forgot to flatten!

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.fc1(x)  # Should be: x = x.view(x.size(0), -1); x = self.fc1(x)
        return x
```

**Problems:**

1. ***
2. ***
3. ***

**Answers:** Missing batch dimension handling, Forgot to flatten conv output, Incorrect fully connected input size

---

### **Challenge 3: Architecture Storytelling**

Complete this story about choosing architectures:

"Once upon a time, Alex had to build three AI systems:

For **recognizing cats in photos**, Alex chose ******\_****** because ******\_******

For **translating between languages**, Alex chose ******\_****** because ******\_******

For **generating creative stories**, Alex chose ******\_****** because ******\_******

Alex learned that choosing the right architecture depends on understanding the ******\_****** and ******\_****** of the problem."

**Sample Answers:** CNN, great with images; LSTM/Transformer, handles sequences well; GPT/RNN, good at generation; data type, problem requirements

---

### **Challenge 4: Performance Optimization**

You're training a deep network but it's too slow. List optimization strategies:

**Data Level:**

1. ***
2. ***

**Model Level:** 3. ************\_************ 4. ************\_************

**Hardware Level:** 5. ************\_************ 6. ************\_************

**Sample Answers:** Data: DataLoader with num_workers, Data augmentation. Model: Smaller model, Mixed precision. Hardware: GPU, Distributed training.

---

## 🎯 **SECTION M: ANSWER KEY & LEARNING TIPS**

### **Understanding Your Mistakes:**

**If you got Architecture Selection wrong:**

- Images → CNN or ViT
- Text sequences → RNN, LSTM, or Transformer
- Generation → GPT or RNN
- Understanding → BERT or Transformer
- Real-time detection → YOLO

**If you got LSTM vs RNN wrong:**

- RNN = Simple memory, forgetful
- LSTM = Smart memory, selective remember/forget

**If you got Attention wrong:**

- Attention = Spotlight on important parts
- Query = What are we looking for?
- Key/Value = What's available?

**If you got Transformer wrong:**

- Transformer = Everyone talks to everyone at once
- Faster and better than RNN for long sequences

### **🎊 CONGRATULATIONS!**

**You've completed all practice questions for Deep Learning!**

### **What Your Scores Mean:**

- **30-37 correct:** 🌟 Deep Learning Expert - Ready for specialization!
- **25-29 correct:** 🎯 Deep Learning Practitioner - Strong foundation
- **20-24 correct:** 📚 Deep Learning Learner - Good understanding
- **Under 20:** 💪 Keep Learning - Study the guide again

### **Learning Path Recommendations:**

**If you scored 30-37:**
✅ You understand deep learning concepts well  
✅ Ready for Step 4: Computer Vision & Image Processing  
✅ Can start building complex AI projects

**If you scored 25-29:**
✅ Strong foundation, some advanced concepts need review  
✅ Focus on Transformers and Attention mechanisms  
✅ Practice with implementation

**If you scored 20-24:**
✅ Good understanding of basics  
✅ Review CNNs, RNNs, and LSTM concepts  
✅ Practice with simpler architectures first

**If you scored under 20:**
✅ Start with neural network fundamentals  
✅ Focus on understanding how each architecture works  
✅ Practice with basic examples before advanced topics

### **Next Steps:**

1. **Implement a simple neural network** from scratch
2. **Experiment with different architectures** on the same dataset
3. **Explain deep learning concepts** to others
4. **Get ready for Step 4: Computer Vision & Image Processing!** 🚀

### **🏆 Final Challenge: AI Architect**

Design a complete AI system for this scenario:

**Scenario:** An autonomous drone that needs to:

- Navigate through forests
- Avoid obstacles
- Find and track animals
- Return to base with photos

**Your complete system design:**

1. **Computer Vision Module:** ******\_******
2. **Navigation Module:** ******\_******
3. **Object Detection Module:** ******\_******
4. **Decision Making Module:** ******\_******

**Explain why you chose each architecture:**

**Sample Solution:**

1. CNN/ViT - Process visual input to understand surroundings
2. CNN + Sensor Fusion - Combine camera + lidar data
3. YOLO - Real-time animal detection and tracking
4. Reinforcement Learning + Rule-based - Make navigation decisions

**Remember: The best AI architects understand when to use which tool!** 🎯
