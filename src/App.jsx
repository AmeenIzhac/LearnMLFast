import { useState, useMemo, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import papers10000 from './papers.json'
import papers1000 from './papers1000.json'
import './App.css'

const PAGE_SIZE = 50

// Venue information with descriptions and prestige ratings
const VENUE_INFO = {
  'Neural Information Processing Systems': {
    abbrev: 'NeurIPS',
    prestige: 'Top-tier',
    description: 'Premier ML conference covering all aspects of neural networks, optimization, deep learning, and computational neuroscience. Historically the most competitive venue.',
    color: '#6366f1'
  },
  'International Conference on Machine Learning': {
    abbrev: 'ICML',
    prestige: 'Top-tier',
    description: 'Flagship ML conference alongside NeurIPS. Strong focus on theoretical foundations, algorithms, and applications of machine learning.',
    color: '#8b5cf6'
  },
  'International Conference on Learning Representations': {
    abbrev: 'ICLR',
    prestige: 'Top-tier',
    description: 'Relatively newer but now top-tier. Known for deep learning innovations, representation learning, and being first to adopt open peer review.',
    color: '#a855f7'
  },
  'Computer Vision and Pattern Recognition': {
    abbrev: 'CVPR',
    prestige: 'Top-tier',
    description: 'The premier computer vision conference. Covers image recognition, object detection, segmentation, 3D vision, and video understanding.',
    color: '#ec4899'
  },
  'IEEE International Conference on Computer Vision': {
    abbrev: 'ICCV',
    prestige: 'Top-tier',
    description: 'Top computer vision venue alongside CVPR. Held biennially with slightly more theoretical focus. Very competitive.',
    color: '#f43f5e'
  },
  'European Conference on Computer Vision': {
    abbrev: 'ECCV',
    prestige: 'Top-tier',
    description: 'Third major CV conference. European-based, held biennially. Known for fundamental vision research and emerging topics.',
    color: '#f97316'
  },
  'Annual Meeting of the Association for Computational Linguistics': {
    abbrev: 'ACL',
    prestige: 'Top-tier',
    description: 'Premier NLP conference. Covers language understanding, generation, machine translation, and computational linguistics.',
    color: '#22c55e'
  },
  'Conference on Empirical Methods in Natural Language Processing': {
    abbrev: 'EMNLP',
    prestige: 'Top-tier',
    description: 'Second major NLP venue. Strong emphasis on empirical, data-driven approaches to language processing.',
    color: '#10b981'
  },
  'North American Chapter of the Association for Computational Linguistics': {
    abbrev: 'NAACL',
    prestige: 'Excellent',
    description: 'Regional ACL chapter but globally competitive. Good venue for NLP research with slightly lower bar than ACL/EMNLP.',
    color: '#14b8a6'
  },
  'AAAI Conference on Artificial Intelligence': {
    abbrev: 'AAAI',
    prestige: 'Excellent',
    description: 'Broad AI conference covering ML, NLP, vision, planning, reasoning, and robotics. One of the oldest AI venues.',
    color: '#0ea5e9'
  }
}

// 100 ML terms across diverse areas and difficulty levels (1-10 scale)
const ALL_CALIBRATION_TERMS = [
  // Fundamentals (difficulty 1-2)
  { term: 'Backpropagation', difficulty: 1, area: 'fundamentals', definition: 'Algorithm that calculates how much each weight in a neural network contributed to the error, working backwards from output to input.' },
  { term: 'Gradient Descent', difficulty: 1, area: 'fundamentals', definition: 'Optimization algorithm that iteratively adjusts parameters by moving in the direction that reduces the error most.' },
  { term: 'Learning Rate', difficulty: 1, area: 'fundamentals', definition: 'Hyperparameter controlling how big of a step the model takes when updating weights during training.' },
  { term: 'Stochastic Gradient Descent', difficulty: 1.5, area: 'fundamentals', definition: 'Variant of gradient descent that updates weights using random subsets (mini-batches) of data instead of the full dataset.' },
  { term: 'Overfitting', difficulty: 1, area: 'fundamentals', definition: 'When a model learns the training data too well, including noise, and fails to generalize to new data.' },
  { term: 'Cross-Validation', difficulty: 1.5, area: 'fundamentals', definition: 'Technique for evaluating models by splitting data into multiple folds and training/testing on different combinations.' },
  { term: 'Regularization', difficulty: 1.5, area: 'fundamentals', definition: 'Techniques that prevent overfitting by adding penalties for model complexity, like L1/L2 weight penalties.' },
  { term: 'Loss Function', difficulty: 1, area: 'fundamentals', definition: 'Mathematical function measuring how wrong the model\'s predictions are compared to the true values.' },
  { term: 'Activation Function', difficulty: 1.5, area: 'fundamentals', definition: 'Non-linear function applied to neuron outputs (like ReLU or sigmoid) that enables networks to learn complex patterns.' },
  { term: 'Softmax', difficulty: 1.5, area: 'fundamentals', definition: 'Function that converts a vector of numbers into probabilities that sum to 1, commonly used for classification.' },

  // Architecture Basics (difficulty 2-3)
  { term: 'Convolutional Neural Network', difficulty: 2, area: 'architectures', definition: 'Neural network using sliding filters to detect local patterns in images, like edges, textures, and shapes.' },
  { term: 'Recurrent Neural Network', difficulty: 2, area: 'architectures', definition: 'Neural network with loops that maintain memory of previous inputs, designed for sequential data like text.' },
  { term: 'Long Short-Term Memory', difficulty: 2.5, area: 'architectures', definition: 'Type of RNN with gates that control information flow, solving the problem of forgetting long-range dependencies.' },
  { term: 'Dropout', difficulty: 2, area: 'training', definition: 'Regularization technique that randomly "turns off" neurons during training to prevent over-reliance on specific pathways.' },
  { term: 'Batch Normalization', difficulty: 2.5, area: 'training', definition: 'Technique that normalizes layer inputs to have zero mean and unit variance, stabilizing and accelerating training.' },
  { term: 'Max Pooling', difficulty: 2, area: 'architectures', definition: 'Operation that reduces spatial dimensions by taking the maximum value in each local region of a feature map.' },
  { term: 'Embedding Layer', difficulty: 2, area: 'architectures', definition: 'Layer that converts discrete tokens (like words) into dense continuous vectors that capture semantic meaning.' },
  { term: 'Encoder-Decoder', difficulty: 2.5, area: 'architectures', definition: 'Architecture where an encoder compresses input into a representation, and a decoder generates output from it.' },
  { term: 'Skip Connection', difficulty: 2.5, area: 'architectures', definition: 'Direct pathway that allows information to bypass one or more layers, helping gradients flow in deep networks.' },
  { term: 'Residual Network', difficulty: 2.5, area: 'architectures', definition: 'Architecture using skip connections where layers learn residual changes rather than full transformations.' },

  // Intermediate Concepts (difficulty 3-4)
  { term: 'Attention Mechanism', difficulty: 3, area: 'nlp', definition: 'Mechanism allowing models to focus on relevant parts of the input when producing each part of the output.' },
  { term: 'Transformer', difficulty: 3, area: 'nlp', definition: 'Architecture based entirely on attention mechanisms, processing all positions in parallel rather than sequentially.' },
  { term: 'Self-Attention', difficulty: 3.5, area: 'nlp', definition: 'Attention mechanism where each position in a sequence attends to all other positions in the same sequence.' },
  { term: 'Positional Encoding', difficulty: 3.5, area: 'nlp', definition: 'Method of injecting position information into transformers since they don\'t inherently understand sequence order.' },
  { term: 'Layer Normalization', difficulty: 3, area: 'training', definition: 'Normalization applied across features for each sample, commonly used in transformers instead of batch norm.' },
  { term: 'Adam Optimizer', difficulty: 3, area: 'training', definition: 'Popular optimizer combining momentum and adaptive learning rates for each parameter.' },
  { term: 'Weight Decay', difficulty: 3, area: 'training', definition: 'Regularization that shrinks weights toward zero during optimization, equivalent to L2 regularization.' },
  { term: 'Transfer Learning', difficulty: 3, area: 'training', definition: 'Using a model pre-trained on one task as a starting point for training on a different but related task.' },
  { term: 'Fine-Tuning', difficulty: 3, area: 'training', definition: 'Continuing to train a pre-trained model on new data, typically with a lower learning rate.' },
  { term: 'Data Augmentation', difficulty: 2.5, area: 'training', definition: 'Artificially expanding training data by applying transformations like rotations, flips, or color changes.' },

  // Generative Models (difficulty 3-5)
  { term: 'Autoencoder', difficulty: 3, area: 'generative', definition: 'Network that learns to compress data into a smaller representation and then reconstruct the original.' },
  { term: 'Variational Autoencoder', difficulty: 4, area: 'generative', definition: 'Autoencoder that learns a probability distribution in latent space, enabling generation of new samples.' },
  { term: 'Generative Adversarial Network', difficulty: 3.5, area: 'generative', definition: 'Two networks (generator and discriminator) trained adversarially - one creates fakes, the other detects them.' },
  { term: 'Latent Space', difficulty: 3.5, area: 'generative', definition: 'Lower-dimensional space where data is represented in compressed form, capturing essential features.' },
  { term: 'KL Divergence', difficulty: 4, area: 'theory', definition: 'Measure of how one probability distribution differs from another, used in VAEs and information theory.' },
  { term: 'Reparameterization Trick', difficulty: 4.5, area: 'generative', definition: 'Technique for backpropagating through random sampling by separating randomness from learned parameters.' },
  { term: 'Mode Collapse', difficulty: 4, area: 'generative', definition: 'GAN failure where the generator only produces a limited variety of outputs instead of the full distribution.' },
  { term: 'Wasserstein Distance', difficulty: 5, area: 'theory', definition: 'Metric measuring the "cost" to transform one distribution into another, used in improved GANs.' },
  { term: 'Flow-Based Models', difficulty: 5, area: 'generative', definition: 'Generative models using invertible transformations to map between data and a simple base distribution.' },
  { term: 'Normalizing Flows', difficulty: 5.5, area: 'generative', definition: 'Sequence of invertible transformations that transform a simple distribution into a complex one.' },

  // NLP Advanced (difficulty 4-6)
  { term: 'BERT', difficulty: 4, area: 'nlp', definition: 'Bidirectional transformer pre-trained on masked language modeling, foundational for NLP transfer learning.' },
  { term: 'GPT', difficulty: 4, area: 'nlp', definition: 'Autoregressive transformer that generates text by predicting the next token, trained on massive text corpora.' },
  { term: 'Masked Language Modeling', difficulty: 4.5, area: 'nlp', definition: 'Pre-training task where model predicts randomly masked tokens in a sentence using bidirectional context.' },
  { term: 'Causal Language Modeling', difficulty: 4.5, area: 'nlp', definition: 'Pre-training task where model predicts the next token using only left context (past tokens).' },
  { term: 'Tokenization', difficulty: 3, area: 'nlp', definition: 'Process of splitting text into smaller units (tokens) that the model can process.' },
  { term: 'Byte-Pair Encoding', difficulty: 4, area: 'nlp', definition: 'Tokenization algorithm that iteratively merges frequent character pairs to build a subword vocabulary.' },
  { term: 'Rotary Position Embedding', difficulty: 6, area: 'nlp', definition: 'Position encoding method that applies rotation matrices to encode relative positions in attention.' },
  { term: 'Flash Attention', difficulty: 6, area: 'nlp', definition: 'Memory-efficient attention algorithm that avoids materializing the full attention matrix, enabling longer sequences.' },
  { term: 'Mixture of Experts', difficulty: 5.5, area: 'architectures', definition: 'Architecture with multiple specialized sub-networks (experts) and a router that selects which to use.' },
  { term: 'Sparse Attention', difficulty: 5.5, area: 'nlp', definition: 'Attention patterns that only attend to a subset of positions, reducing computational cost.' },

  // Computer Vision (difficulty 3-6)
  { term: 'Feature Pyramid Network', difficulty: 4, area: 'vision', definition: 'Architecture that combines features at multiple scales for detecting objects of different sizes.' },
  { term: 'Non-Maximum Suppression', difficulty: 4, area: 'vision', definition: 'Algorithm that removes redundant overlapping bounding boxes, keeping only the best detection.' },
  { term: 'Anchor Boxes', difficulty: 4, area: 'vision', definition: 'Pre-defined boxes of various sizes/ratios that object detectors use as reference for predictions.' },
  { term: 'Vision Transformer', difficulty: 4.5, area: 'vision', definition: 'Transformer architecture applied to images by treating image patches as tokens.' },
  { term: 'CLIP', difficulty: 5, area: 'vision', definition: 'Model trained to align images and text in a shared embedding space using contrastive learning.' },
  { term: 'Patch Embedding', difficulty: 4.5, area: 'vision', definition: 'Converting image patches into vectors that can be processed by transformers.' },
  { term: 'Semantic Segmentation', difficulty: 3.5, area: 'vision', definition: 'Task of classifying every pixel in an image into semantic categories.' },
  { term: 'Instance Segmentation', difficulty: 4, area: 'vision', definition: 'Task of detecting and delineating each object instance with pixel-level masks.' },
  { term: 'Depth Estimation', difficulty: 4, area: 'vision', definition: 'Task of predicting the distance of each pixel from the camera.' },
  { term: 'Optical Flow', difficulty: 4.5, area: 'vision', definition: 'The pattern of apparent motion between consecutive frames, representing pixel displacement.' },

  // Self-Supervised & Contrastive (difficulty 5-7)
  { term: 'Contrastive Learning', difficulty: 5, area: 'self-supervised', definition: 'Learning representations by pulling similar samples together and pushing dissimilar ones apart in embedding space.' },
  { term: 'SimCLR', difficulty: 5.5, area: 'self-supervised', definition: 'Contrastive learning framework that creates positive pairs via data augmentation of the same image.' },
  { term: 'InfoNCE Loss', difficulty: 6, area: 'self-supervised', definition: 'Contrastive loss function that maximizes mutual information between positive pairs relative to negatives.' },
  { term: 'Momentum Contrast', difficulty: 5.5, area: 'self-supervised', definition: 'Contrastive method using a momentum-updated encoder to maintain a large consistent queue of negatives.' },
  { term: 'BYOL', difficulty: 6, area: 'self-supervised', definition: 'Self-supervised method that learns without negative pairs using two networks with asymmetric architecture.' },
  { term: 'DINO', difficulty: 6, area: 'self-supervised', definition: 'Self-distillation method where a student network learns from a momentum teacher, emerging semantic features.' },
  { term: 'Masked Autoencoder', difficulty: 5.5, area: 'self-supervised', definition: 'Pre-training method that masks large portions of an image and trains to reconstruct the missing parts.' },
  { term: 'Self-Distillation', difficulty: 6, area: 'self-supervised', definition: 'Training a student network to match outputs of a teacher that is a copy of itself.' },

  // Reinforcement Learning (difficulty 4-7)
  { term: 'Q-Learning', difficulty: 4, area: 'rl', definition: 'RL algorithm that learns action-value function estimating expected reward for each state-action pair.' },
  { term: 'Policy Gradient', difficulty: 4.5, area: 'rl', definition: 'RL methods that directly optimize the policy by estimating gradients of expected reward.' },
  { term: 'Actor-Critic', difficulty: 5, area: 'rl', definition: 'RL architecture combining policy learning (actor) with value estimation (critic) for lower variance.' },
  { term: 'Proximal Policy Optimization', difficulty: 5.5, area: 'rl', definition: 'Policy gradient algorithm using clipped objectives for stable and efficient training.' },
  { term: 'Reward Shaping', difficulty: 5, area: 'rl', definition: 'Modifying the reward signal to guide learning, such as adding intermediate rewards.' },
  { term: 'RLHF', difficulty: 6, area: 'rl', definition: 'Reinforcement Learning from Human Feedback - training models using human preference comparisons as reward.' },
  { term: 'Constitutional AI', difficulty: 6.5, area: 'rl', definition: 'Training AI to follow principles by having it critique and revise its own outputs.' },
  { term: 'Direct Preference Optimization', difficulty: 7, area: 'rl', definition: 'Method that optimizes language models from preferences without explicit reward modeling.' },

  // Diffusion & Modern Generative (difficulty 5-8)
  { term: 'Diffusion Model', difficulty: 5, area: 'generative', definition: 'Generative model that learns to reverse a gradual noising process, generating samples from noise.' },
  { term: 'Score Matching', difficulty: 6, area: 'generative', definition: 'Training method that learns the gradient of the log probability density (score function).' },
  { term: 'Denoising Score Matching', difficulty: 6.5, area: 'generative', definition: 'Learning score function by training to denoise samples corrupted with known noise.' },
  { term: 'DDPM', difficulty: 6, area: 'generative', definition: 'Denoising Diffusion Probabilistic Models - foundational diffusion approach with fixed noise schedule.' },
  { term: 'Classifier-Free Guidance', difficulty: 6.5, area: 'generative', definition: 'Technique that improves conditional generation by mixing conditional and unconditional model predictions.' },
  { term: 'Latent Diffusion', difficulty: 6, area: 'generative', definition: 'Diffusion process applied in a compressed latent space rather than pixel space for efficiency.' },
  { term: 'Stable Diffusion', difficulty: 5.5, area: 'generative', definition: 'Popular open-source text-to-image model using latent diffusion with CLIP text conditioning.' },
  { term: 'Rectified Flow', difficulty: 7.5, area: 'generative', definition: 'Generative approach learning straight paths between noise and data distributions.' },

  // 3D & Neural Rendering (difficulty 6-9)
  { term: 'Neural Radiance Field', difficulty: 7, area: '3d', definition: 'Represents 3D scenes as neural networks predicting color and density at any point, enabling novel view synthesis.' },
  { term: 'Volume Rendering', difficulty: 6.5, area: '3d', definition: 'Rendering technique that integrates color/density along rays to produce 2D images from 3D representations.' },
  { term: 'Signed Distance Function', difficulty: 6, area: '3d', definition: 'Function giving the distance to the nearest surface, with sign indicating inside/outside.' },
  { term: 'Gaussian Splatting', difficulty: 7.5, area: '3d', definition: 'Represents scenes as collections of 3D Gaussians that are "splatted" onto the image plane for fast rendering.' },
  { term: 'Implicit Neural Representation', difficulty: 7, area: '3d', definition: 'Representing signals (images, 3D shapes) as continuous functions parameterized by neural networks.' },
  { term: 'Photometric Loss', difficulty: 5.5, area: '3d', definition: 'Loss measuring difference between rendered and observed images, used to train view synthesis models.' },
  { term: 'Multi-View Stereo', difficulty: 6, area: '3d', definition: 'Reconstructing 3D geometry by finding correspondences across multiple calibrated images.' },
  { term: 'Bundle Adjustment', difficulty: 7, area: '3d', definition: 'Joint optimization of 3D structure and camera parameters to minimize reprojection error.' },

  // Theory & Foundations (difficulty 6-10)
  { term: 'PAC Learning', difficulty: 7, area: 'theory', definition: 'Probably Approximately Correct - framework for analyzing when learning algorithms can succeed with high probability.' },
  { term: 'VC Dimension', difficulty: 7.5, area: 'theory', definition: 'Measure of model capacity - the largest set of points that can be shattered (perfectly classified) by the model.' },
  { term: 'Rademacher Complexity', difficulty: 8, area: 'theory', definition: 'Measure of how well a function class can fit random noise, used to bound generalization error.' },
  { term: 'Neural Tangent Kernel', difficulty: 8, area: 'theory', definition: 'Kernel that describes neural network training dynamics in the infinite-width limit.' },
  { term: 'Double Descent', difficulty: 7, area: 'theory', definition: 'Phenomenon where test error decreases, increases, then decreases again as model complexity grows past interpolation.' },
  { term: 'Lottery Ticket Hypothesis', difficulty: 7, area: 'theory', definition: 'Claim that dense networks contain sparse subnetworks that can achieve the same performance when trained in isolation.' },
  { term: 'Grokking', difficulty: 8, area: 'theory', definition: 'Phenomenon where models suddenly generalize long after memorizing training data, showing delayed understanding.' },
  { term: 'Scaling Laws', difficulty: 6.5, area: 'theory', definition: 'Empirical relationships showing how model performance improves predictably with compute, data, and parameters.' },

  // Cutting Edge (difficulty 8-10)
  { term: 'Mechanistic Interpretability', difficulty: 8.5, area: 'interpretability', definition: 'Reverse-engineering neural networks to understand the algorithms and features they have learned.' },
  { term: 'Superposition Hypothesis', difficulty: 9, area: 'interpretability', definition: 'Theory that neural networks represent more features than neurons by encoding them in overlapping directions.' },
  { term: 'Circuit Analysis', difficulty: 8.5, area: 'interpretability', definition: 'Identifying and understanding the computational subgraphs (circuits) in neural networks that perform specific tasks.' },
  { term: 'Induction Heads', difficulty: 9, area: 'interpretability', definition: 'Attention pattern in transformers that copies tokens appearing after similar prior contexts, enabling in-context learning.' },
  { term: 'In-Context Learning', difficulty: 7, area: 'llm', definition: 'Ability of large language models to learn tasks from examples in the prompt without weight updates.' },
  { term: 'Chain-of-Thought', difficulty: 6, area: 'llm', definition: 'Prompting technique where models show reasoning steps, improving performance on complex tasks.' },
  { term: 'Emergent Abilities', difficulty: 7.5, area: 'llm', definition: 'Capabilities that appear suddenly at scale that were not present in smaller models.' },
  { term: 'State Space Models', difficulty: 8, area: 'architectures', definition: 'Sequence models using linear state transitions with selective mechanisms, alternative to transformers.' },
  { term: 'Mamba Architecture', difficulty: 8.5, area: 'architectures', definition: 'Selective state space model with input-dependent parameters enabling efficient long-range modeling.' },
  { term: 'Kolmogorov-Arnold Networks', difficulty: 9.5, area: 'architectures', definition: 'Networks based on Kolmogorov-Arnold representation theorem, using learnable univariate functions on edges.' },
]

// Helper to select next term based on adaptive difficulty
function selectNextTerm(usedTerms, currentDifficulty, allTerms) {
  const available = allTerms.filter(t => !usedTerms.includes(t.term))
  if (available.length === 0) return null

  // Find terms closest to current difficulty, with some randomness for diversity
  const scored = available.map(t => ({
    ...t,
    score: Math.abs(t.difficulty - currentDifficulty) + Math.random() * 0.5
  }))
  scored.sort((a, b) => a.score - b.score)

  // Pick from top 5 closest matches for variety
  const candidates = scored.slice(0, Math.min(5, scored.length))
  return candidates[Math.floor(Math.random() * candidates.length)]
}

async function makeIntuitive(title, abstract, calibrationData, misunderstandingSummaries = [], onChunk = null) {
  // Build context from actual term ratings
  const knowledgeContext = calibrationData
    .map(({ term, rating }) => `- "${term}": ${rating}/10 familiarity`)
    .join('\n')

  // Identify low-familiarity concepts (rating <= 4) to flag for explicit explanation
  const lowFamiliarity = calibrationData
    .filter(({ rating }) => rating <= 4)
    .map(({ term }) => term)

  const highFamiliarity = calibrationData
    .filter(({ rating }) => rating >= 7)
    .map(({ term }) => term)

  // Build misunderstanding context from accumulated session data
  const misunderstandingContext = misunderstandingSummaries.length > 0
    ? `\n\nPREVIOUS MISUNDERSTANDINGS (from this session's clarification chats):\n${misunderstandingSummaries.map((s, i) => `${i + 1}. ${s}`).join('\n')}\n\nUse these insights to provide even clearer explanations - pay special attention to concepts or phrasings that confused them before.`
    : ''

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{
        role: 'user',
        content: `You are helping someone understand a research paper. Here is their self-reported familiarity with various ML concepts:

${knowledgeContext}

CONCEPTS THEY LIKELY DON'T KNOW (rated ≤4): ${lowFamiliarity.length > 0 ? lowFamiliarity.join(', ') : 'None identified'}
CONCEPTS THEY KNOW WELL (rated ≥7): ${highFamiliarity.length > 0 ? highFamiliarity.join(', ') : 'None identified'}${misunderstandingContext}

Your task: Rewrite this paper's title and abstract for this reader.

IMPORTANT GUIDELINES:
1. When the paper uses a concept the reader likely doesn't know, EXPLICITLY explain what it means in parentheses or with a brief clarification. Don't assume they'll understand technical terms or implied meanings.
2. For concepts the reader knows well, you can use them directly without explanation.
3. Be specific - don't say things like "learning differences instead of new functions" without explaining what "functions" means in this context if they wouldn't know.
4. Use concrete analogies for unfamiliar concepts.
5. Don't over-explain concepts they already understand well.

TITLE: ${title}

ABSTRACT: ${abstract || 'No abstract available.'}

Respond in this exact format:
SIMPLE TITLE: [your simplified title]
SIMPLE ABSTRACT: [your simplified abstract, 3-4 sentences max, with explicit explanations where needed]`
      }],
      max_tokens: 500,
      stream: true
    })
  })

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let fullContent = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const chunk = decoder.decode(value, { stream: true })
    const lines = chunk.split('\n').filter(line => line.trim() !== '')

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6)
        if (data === '[DONE]') continue

        try {
          const parsed = JSON.parse(data)
          const content = parsed.choices?.[0]?.delta?.content
          if (content) {
            fullContent += content
            // Parse current content to extract title and abstract so far
            if (onChunk) {
              const titleMatch = fullContent.match(/SIMPLE TITLE:\s*(.+?)(?=SIMPLE ABSTRACT:|$)/is)
              const abstractMatch = fullContent.match(/SIMPLE ABSTRACT:\s*(.+)/is)
              onChunk({
                title: titleMatch ? titleMatch[1].trim() : '',
                abstract: abstractMatch ? abstractMatch[1].trim() : ''
              })
            }
          }
        } catch (e) {
          // Skip malformed chunks
        }
      }
    }
  }

  const titleMatch = fullContent.match(/SIMPLE TITLE:\s*(.+?)(?=SIMPLE ABSTRACT:|$)/is)
  const abstractMatch = fullContent.match(/SIMPLE ABSTRACT:\s*(.+)/is)

  return {
    title: titleMatch ? titleMatch[1].trim() : title,
    abstract: abstractMatch ? abstractMatch[1].trim() : abstract
  }
}

// Chat about abstract - for clarification questions (with streaming)
async function chatAboutAbstract(paper, simplified, chatHistory, userMessage, onChunk) {
  const messages = [
    {
      role: 'system',
      content: `You are a helpful research paper explainer. The user is reading this paper:

TITLE: ${paper.title}
ABSTRACT: ${paper.abstract || 'No abstract available.'}

You previously simplified it as:
SIMPLIFIED TITLE: ${simplified.title}
SIMPLIFIED ABSTRACT: ${simplified.abstract}

Answer the user's clarification questions concisely. Be helpful and educational. If they seem confused about something, explain it clearly with analogies if helpful.`
    },
    ...chatHistory.map(msg => ({
      role: msg.role,
      content: msg.content
    })),
    { role: 'user', content: userMessage }
  ]

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages,
      max_tokens: 400,
      stream: true
    })
  })

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let fullContent = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const chunk = decoder.decode(value, { stream: true })
    const lines = chunk.split('\n').filter(line => line.trim() !== '')

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6)
        if (data === '[DONE]') continue

        try {
          const parsed = JSON.parse(data)
          const content = parsed.choices?.[0]?.delta?.content
          if (content) {
            fullContent += content
            onChunk(fullContent)
          }
        } catch (e) {
          // Skip malformed chunks
        }
      }
    }
  }

  return fullContent
}

// Summarize misunderstandings from a chat session
async function summarizeMisunderstandings(paper, chatHistory) {
  if (chatHistory.length < 2) return null // Need at least one Q&A pair

  const chatTranscript = chatHistory
    .map(msg => `${msg.role === 'user' ? 'USER' : 'ASSISTANT'}: ${msg.content}`)
    .join('\n\n')

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{
        role: 'user',
        content: `Analyze this clarification chat about a research paper abstract. Extract what concepts or phrasings the user struggled to understand.

PAPER TITLE: ${paper.title}

CHAT TRANSCRIPT:
${chatTranscript}

Provide a CONCISE summary (1-2 sentences max) of what the user found confusing or needed clarification on. This will be used to improve future explanations. Focus on the underlying concepts they struggled with, not the specific questions they asked.

If the chat was just casual or the user understood everything fine, respond with just: "NO_ISSUES"

Summary:`
      }],
      max_tokens: 150
    })
  })

  const data = await response.json()
  const summary = data.choices[0].message.content.trim()

  if (summary === 'NO_ISSUES' || summary.includes('NO_ISSUES')) {
    return null
  }

  return summary
}

function CalibrationModal({ onComplete, onClose }) {
  const [step, setStep] = useState(0)
  const [ratings, setRatings] = useState([])
  const [currentTerm, setCurrentTerm] = useState(null)
  const [currentRating, setCurrentRating] = useState(5)
  const [currentDifficulty, setCurrentDifficulty] = useState(5)
  const [usedTerms, setUsedTerms] = useState([])
  const [showDefinition, setShowDefinition] = useState(false)

  const TOTAL_STEPS = 10

  useEffect(() => {
    // Start with a medium difficulty term
    const firstTerm = selectNextTerm([], 5, ALL_CALIBRATION_TERMS)
    setCurrentTerm(firstTerm)
  }, [])

  const handleNext = () => {
    // Record current rating
    const newRating = { term: currentTerm.term, rating: currentRating, difficulty: currentTerm.difficulty }
    const newRatings = [...ratings, newRating]
    setRatings(newRatings)

    const newUsedTerms = [...usedTerms, currentTerm.term]
    setUsedTerms(newUsedTerms)

    if (step + 1 >= TOTAL_STEPS) {
      // Done - pass the raw ratings data
      onComplete(newRatings)
      return
    }

    // Adjust difficulty based on rating
    // High rating (8-10) -> increase difficulty
    // Low rating (1-3) -> decrease difficulty
    // Medium (4-7) -> slight adjustment toward rating
    let newDifficulty = currentDifficulty
    if (currentRating >= 8) {
      newDifficulty = Math.min(10, currentDifficulty + 1.5)
    } else if (currentRating <= 3) {
      newDifficulty = Math.max(1, currentDifficulty - 1.5)
    } else if (currentRating >= 6) {
      newDifficulty = Math.min(10, currentDifficulty + 0.5)
    } else if (currentRating <= 4) {
      newDifficulty = Math.max(1, currentDifficulty - 0.5)
    }

    setCurrentDifficulty(newDifficulty)

    // Select next term based on new difficulty
    const nextTerm = selectNextTerm(newUsedTerms, newDifficulty, ALL_CALIBRATION_TERMS)
    setCurrentTerm(nextTerm)
    setCurrentRating(5) // Reset slider
    setShowDefinition(false) // Reset definition visibility
    setStep(step + 1)
  }

  if (!currentTerm) return null

  return (
    <div className="modal-overlay">
      <div className="modal calibration-modal">
        <h2>Calibrate Your Knowledge</h2>
        <p className="modal-subtitle">
          Rate your familiarity with this term ({step + 1}/{TOTAL_STEPS})
        </p>

        <div className="calibration-progress">
          <div
            className="progress-bar"
            style={{ width: `${((step + 1) / TOTAL_STEPS) * 100}%` }}
          />
        </div>

        <div className="single-term">
          <h3 className="term-name-large">{currentTerm.term}</h3>
          <p className="term-area">{currentTerm.area.replace('-', ' ')}</p>

          {showDefinition ? (
            <p className="term-definition">{currentTerm.definition}</p>
          ) : (
            <button
              className="reveal-definition-btn"
              onClick={() => setShowDefinition(true)}
            >
              🔍 Reveal Definition
            </button>
          )}
        </div>

        <div className="rating-section">
          <div className="rating-labels">
            <span>Never heard of it</span>
            <span>Expert</span>
          </div>
          <input
            type="range"
            min="1"
            max="10"
            value={currentRating}
            onChange={(e) => setCurrentRating(parseInt(e.target.value))}
            className="rating-slider"
          />
          <div className="rating-value-large">{currentRating}</div>
        </div>

        <div className="modal-actions">
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={handleNext}>
            {step + 1 >= TOTAL_STEPS ? 'Finish' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  )
}

function IntuitiveModal({ paper, calibrationData, misunderstandingSummaries, onClose, onAddMisunderstanding }) {
  const [loading, setLoading] = useState(true)
  const [simplified, setSimplified] = useState(null)
  const [streamingSimplified, setStreamingSimplified] = useState({ title: '', abstract: '' })
  const [error, setError] = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [showChat, setShowChat] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const chatEndRef = useRef(null)

  useEffect(() => {
    makeIntuitive(
      paper.title,
      paper.abstract,
      calibrationData,
      misunderstandingSummaries,
      (partialResult) => {
        setStreamingSimplified(partialResult)
      }
    )
      .then(result => {
        setSimplified(result)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [paper, calibrationData, misunderstandingSummaries])

  useEffect(() => {
    // Auto-scroll chat to bottom
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [chatHistory, streamingContent])

  const handleSendMessage = async () => {
    if (!chatInput.trim() || chatLoading) return

    const userMessage = chatInput.trim()
    setChatInput('')
    setChatHistory(prev => [...prev, { role: 'user', content: userMessage }])
    setChatLoading(true)
    setStreamingContent('')

    try {
      const finalContent = await chatAboutAbstract(
        paper,
        simplified,
        chatHistory,
        userMessage,
        (partialContent) => {
          setStreamingContent(partialContent)
        }
      )
      // Add the complete message to history
      setChatHistory(prev => [...prev, { role: 'assistant', content: finalContent }])
      setStreamingContent('')
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }])
      setStreamingContent('')
    }

    setChatLoading(false)
  }

  const handleClose = async () => {
    // If there was chat, summarize misunderstandings
    if (chatHistory.length >= 2) {
      try {
        const summary = await summarizeMisunderstandings(paper, chatHistory)
        if (summary && onAddMisunderstanding) {
          onAddMisunderstanding(summary)
        }
      } catch (err) {
        console.error('Failed to summarize misunderstandings:', err)
      }
    }
    onClose()
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="modal-overlay">
      <div className="modal intuitive-modal">
        <h2>Simplified Version</h2>

        {loading && !streamingSimplified.title && <div className="loading">Simplifying based on your knowledge profile...</div>}

        {error && <div className="error">Error: {error}</div>}

        {(simplified || streamingSimplified.title) && (
          <>
            <div className="simplified-section">
              <label>Original Title</label>
              <p className="original">{paper.title}</p>
            </div>
            <div className="simplified-section">
              <label>Simplified Title {loading && <span className="streaming-indicator">●</span>}</label>
              <p className={`simplified ${loading ? 'streaming' : ''}`}>{simplified?.title || streamingSimplified.title}</p>
            </div>
            <div className="simplified-section">
              <label>Simplified Abstract {loading && <span className="streaming-indicator">●</span>}</label>
              <p className={`simplified ${loading ? 'streaming' : ''}`}>{simplified?.abstract || streamingSimplified.abstract || (loading ? 'Generating...' : '')}</p>
            </div>

            {/* Chat Section */}
            <div className="chat-section">
              {!showChat ? (
                <button
                  className="start-chat-btn"
                  onClick={() => setShowChat(true)}
                >
                  💬 Ask for clarification
                </button>
              ) : (
                <>
                  <div className="chat-header">
                    <span>Ask questions about this paper</span>
                  </div>
                  <div className="chat-messages">
                    {chatHistory.length === 0 && (
                      <div className="chat-hint">
                        Ask anything you don't understand about the abstract!
                      </div>
                    )}
                    {chatHistory.map((msg, idx) => (
                      <div key={idx} className={`chat-message ${msg.role}`}>
                        <div className="message-content">{msg.content}</div>
                      </div>
                    ))}
                    {chatLoading && (
                      <div className="chat-message assistant">
                        <div className={`message-content ${streamingContent ? 'streaming' : 'typing'}`}>
                          {streamingContent || 'Thinking...'}
                        </div>
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </div>
                  <div className="chat-input-area">
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Type your question..."
                      disabled={chatLoading}
                    />
                    <button
                      onClick={handleSendMessage}
                      disabled={chatLoading || !chatInput.trim()}
                    >
                      Send
                    </button>
                  </div>
                </>
              )}
            </div>
          </>
        )}

        <div className="modal-actions">
          <button className="btn-primary" onClick={handleClose}>Close</button>
        </div>
      </div>
    </div>
  )
}

function App() {
  const navigate = useNavigate()
  const [dataset, setDataset] = useState('10000')
  const [sortBy, setSortBy] = useState('citations')
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE)
  const [calibrationData, setCalibrationData] = useState(() => {
    const saved = localStorage.getItem('calibrationData')
    return saved ? JSON.parse(saved) : null
  })
  const [showCalibration, setShowCalibration] = useState(() => {
    // Automatically show calibration if no saved data exists
    const saved = localStorage.getItem('calibrationData')
    return !saved
  })
  const [intuitiveTarget, setIntuitiveTarget] = useState(null)
  const [expandedPaperId, setExpandedPaperId] = useState(null)
  const [selectedVenue, setSelectedVenue] = useState(null)
  const [showVenueInfo, setShowVenueInfo] = useState(false)
  // Session-level accumulated misunderstanding summaries (not persisted)
  const [misunderstandingSummaries, setMisunderstandingSummaries] = useState([])

  const handleAddMisunderstanding = (summary) => {
    setMisunderstandingSummaries(prev => [...prev, summary])
    console.log('Added misunderstanding summary:', summary)
  }

  const papers = useMemo(() => {
    return dataset === '10000' ? papers10000 : papers1000
  }, [dataset])

  // Get unique venues from papers
  const venues = useMemo(() => {
    const uniqueVenues = [...new Set(papers.map(p => p.venue).filter(Boolean))]
    return uniqueVenues.sort((a, b) => {
      const infoA = VENUE_INFO[a]
      const infoB = VENUE_INFO[b]
      if (!infoA) return 1
      if (!infoB) return -1
      return infoA.abbrev.localeCompare(infoB.abbrev)
    })
  }, [papers])

  const filteredAndSortedPapers = useMemo(() => {
    let filtered = [...papers]

    // Filter by venue if selected
    if (selectedVenue) {
      filtered = filtered.filter(p => p.venue === selectedVenue)
    }

    // Sort
    if (sortBy === 'citations') {
      filtered.sort((a, b) => b.citationCount - a.citationCount)
    } else if (sortBy === 'date-new') {
      filtered.sort((a, b) => b.year - a.year)
    }
    return filtered
  }, [papers, sortBy, selectedVenue])

  const visiblePapers = filteredAndSortedPapers.slice(0, visibleCount)
  const hasMore = visibleCount < filteredAndSortedPapers.length

  const handleDatasetChange = (newDataset) => {
    setDataset(newDataset)
    setVisibleCount(PAGE_SIZE)
  }

  const handleSortChange = (newSort) => {
    setSortBy(newSort)
    setVisibleCount(PAGE_SIZE)
  }

  const handleVenueChange = (venue) => {
    setSelectedVenue(venue === selectedVenue ? null : venue)
    setVisibleCount(PAGE_SIZE)
  }

  const handleCalibrationComplete = (data) => {
    setCalibrationData(data)
    localStorage.setItem('calibrationData', JSON.stringify(data))
    setShowCalibration(false)
  }

  // Calculate average rating for display
  const avgRating = calibrationData
    ? Math.round(calibrationData.reduce((sum, r) => sum + r.rating, 0) / calibrationData.length)
    : null

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>Research Papers Explained at Your Level</h1>
          <p className="tagline">Discover the most influential papers in AI & ML</p>
        </div>
      </header>

      <div className="controls">
        <div className="control-group">
          <label>Dataset</label>
          <div className="button-group">
            <button
              className={dataset === '10000' ? 'active' : ''}
              onClick={() => handleDatasetChange('10000')}
            >
              10,000+ Citations
            </button>
            <button
              className={dataset === '1000' ? 'active' : ''}
              onClick={() => handleDatasetChange('1000')}
            >
              1,000+ Citations
            </button>
          </div>
        </div>

        <div className="control-group">
          <label>Sort by</label>
          <div className="button-group">
            <button
              className={sortBy === 'citations' ? 'active' : ''}
              onClick={() => handleSortChange('citations')}
            >
              Citations
            </button>
            <button
              className={sortBy === 'date-new' ? 'active' : ''}
              onClick={() => handleSortChange('date-new')}
            >
              Newest
            </button>
          </div>
        </div>

        <div className="control-group">
          <label>Calibration</label>
          <div className="button-group">
            <button
              className={calibrationData ? 'active' : ''}
              onClick={() => setShowCalibration(true)}
            >
              {calibrationData ? `Avg: ${avgRating}/10` : 'Calibrate'}
            </button>
          </div>
        </div>

        <div className="stats">
          <div className="stat">
            <span className="stat-value">{filteredAndSortedPapers.length.toLocaleString()}</span>
            <span className="stat-label">{selectedVenue ? 'matching' : 'papers'}</span>
          </div>
        </div>
      </div>

      {/* Venue Filter */}
      <div className="venue-filter">
        <div className="venue-filter-header">
          <span className="venue-filter-label">Filter by Venue</span>
        </div>
        <div className="venue-pills">
          <button
            className={`venue-pill ${!selectedVenue ? 'active' : ''}`}
            onClick={() => handleVenueChange(null)}
          >
            All Venues
          </button>
          {venues.map(venue => {
            const info = VENUE_INFO[venue]
            return (
              <button
                key={venue}
                className={`venue-pill ${selectedVenue === venue ? 'active' : ''}`}
                onClick={() => handleVenueChange(venue)}
                style={selectedVenue === venue && info ? { background: info.color, borderColor: info.color } : {}}
              >
                {info ? info.abbrev : venue}
                {info && <span className="prestige-badge">{info.prestige}</span>}
              </button>
            )
          })}
        </div>
        {showVenueInfo && selectedVenue && VENUE_INFO[selectedVenue] && (
          <div className="venue-description">
            <strong>{VENUE_INFO[selectedVenue].abbrev}</strong> — {selectedVenue}
            <p>{VENUE_INFO[selectedVenue].description}</p>
          </div>
        )}
      </div>

      <main className="papers-grid" key={`${dataset}-${sortBy}-${selectedVenue || 'all'}`}>
        {visiblePapers.map((paper, index) => (
          <article
            key={paper.paperId}
            className="paper-card clickable"
            onClick={() => {
              if (calibrationData) {
                navigate(`/paper/${paper.paperId}`)
              } else {
                setShowCalibration(true)
              }
            }}
          >
            <div className="card-header">
              <span className="rank">#{index + 1}</span>
              <span className="year">{paper.year}</span>
            </div>

            <h2 className="title">{paper.title}</h2>

            <div className="venue">{paper.venue}</div>

            <div className="abstract">
              <p>{paper.abstract || 'Abstract not available.'}</p>
            </div>

            <div className="card-footer">
              <div className="authors">
                {paper.authors.slice(0, 3).map(a => a.name).join(', ')}
                {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
              </div>
              <div className="card-actions">
                <div className="citations">
                  <span>{paper.citationCount.toLocaleString()} citations</span>
                </div>
              </div>
            </div>
          </article>
        ))}
      </main>

      {hasMore && (
        <div className="load-more">
          <button onClick={() => setVisibleCount(v => v + PAGE_SIZE)}>
            Load More ({filteredAndSortedPapers.length - visibleCount} remaining)
          </button>
        </div>
      )}

      {showCalibration && (
        <CalibrationModal
          onComplete={handleCalibrationComplete}
          onClose={() => setShowCalibration(false)}
        />
      )}

      {intuitiveTarget && calibrationData && (
        <IntuitiveModal
          paper={intuitiveTarget}
          calibrationData={calibrationData}
          misunderstandingSummaries={misunderstandingSummaries}
          onClose={() => setIntuitiveTarget(null)}
          onAddMisunderstanding={handleAddMisunderstanding}
        />
      )}
    </div>
  )
}

export default App
