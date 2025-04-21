论断

- Declarative sentences: e.g., The killer was younger than the victim.  
    • Commonsense: e.g., a father cannot be younger than his child; a parent and his or  
    her child cannot be twins.  
    • Inference rules: e.g., if x, then ...... we have a contradiction. Therefore, x must not  
    be true.  
    

# Propositions & Connectives

Propositional logic is based on propositions.  
• Proposition (命题): Aproposition is a declarative sentence that can be judged as  
either true or false.  
• Atomic Proposition (原子命题): Apropositionthat does not contain any smaller  
part that is still a proposition is called an atomic proposition.  

CompoundProposition (复合命题): Apropositionthat involves the assembly of multiple  
propositions is called a compound proposition.  
Words that connect multiple propositions to form a compound proposition are called  
logical connectives. For example:  
• ... and ... (并且) $\wedge$  
• not... (并非) $\neg$  
• ... or ... (或者) $\vee$  
• if ... then ... (如果... 那么...) $\rightarrow$  
• ... if and only if ... (当且仅当) $\leftrightarrow$

组成部件

LP,asthe language of proposition logic, has types of symbols.  
• Atomic proposition (Atom): p,q,r,...p1,r2,...  
• Logical connectives: ∧,∨,¬,→,↔  
• Punctuation: ( and )  

- The length of an expression is the number of occurrences of symbols in it.  
    • Two expressions u and v are equal if they are of the same length and have the same  
    symbols in the same order.  
    

合法形式

The well-formed formulas (wff) of propositional logic are expressions which we obtain by  
using the construction rules below:  
• Every atom (e.g., p) is a well-formed formula.  
• If αis awell-formed formula, then so is (¬α).  
• If αandβ arewell-formed formulas, then so is (α ∧ β), (α ∨ β), (α → β), and  
(α ↔β). (braces matter!)  
• Nothing else is a well-formed formula.  

定义

Definition . Atom(LP)  
Atom(LP) is the set of expressions of LP consisting of an atom proposition symbol only.  

Definition . Form(LP)  
Form(LP) is the smallest set of expressions that satisfies (i)∼(iii):  
(i) Atom(LP) ⊆ Form(LP)  
(ii) If α ∈ Form(LP), then (¬α) ∈ Form(LP).  
(iii) If α, β ∈ Form(LP), then (α ∧ β), (α ∨ β), (α → β), (α ↔ β)∈Form(LP).  

通用性质

Lemma 1.  
Well-formed formulas of LP are non-empty expressions.  

Lemma 2.  
Every well-formed formula of LP has an equal number of opening and closing brackets  

Lemma3 .  
• Every proper prefix of a well-formed formula in LP has more opening brackets than  
closing brackets.  
• Similarly, every proper suffix of a well-formed formula in LP has more closing  
brackets than opening brackets.  
• Hence, proper prefix and proper suffix are not wff in LP (Lemma 2. ).  

Parse Tree解析树

Definition 3.3  
A parse tree of a formula in LP is a tree such that  
• The root is the formula.  
• Leaves are atoms, and  
• Each internal node is formed by applying some formation rule on its children.  

Theorem 3.1  
An expression of LP is a well-formed formula if and only if there is a parse tree of it.  

Definition 3.4  
A formula G is a subformula of formula F if G occurs within F. G is a proper subformula of  
F if G≠ F.  
The nodes of the parse tree of F form the set of subformulas of F.  

Definition 3.5  
Immediate subformulas are the children of a formula in its parse tree, and leading  
connective is the connective that is used to join the children.  

Wecansimplify a parse tree by highlighting only the leading connectives and the atom  
propositions (leaves).  

  

  

Unique Readability Theorem  
There is a unique way to construct every well-formed formula.