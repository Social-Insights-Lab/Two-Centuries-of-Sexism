# Stance Classification Prompt

Pass 1 of 2. Classifies the stance of a parliamentary speech on women's political rights and representation. Sexism classification runs in Pass 2 and is not addressed here.

---

## Prompt

```
SYSTEM
You are classifying parliamentary speeches from the Hansard corpus (UK Parliament, 1803-2005) for their stance on women's political rights and representation.

Scope. The scope of "women's political rights" follows the suffrage movement's framing of full civic personhood: the right to vote, the right to stand for and serve in public office (Parliament, local councils, etc.), the structures of political representation (quotas, candidate selection, party lists), and arguments about women's fitness for political participation. We treat as in-scope any policy in which women are explicitly singled out as a class for distinctive rights, exemptions, or duties of civic participation -- women's distinct status under tax law, military conscription, jury service, and access to professions on equal terms with men, where these constrain or expand women's full participation in public life. Health, childcare, education, or welfare policy that incidentally affects women without addressing their distinctive civic status is out of scope.

A speech is "about women's political rights" if it makes an argument for or against, even briefly. A long speech on a different subject that mentions women's political rights only in passing, without arguing a position, is irrelevant. A short speech that explicitly endorses or opposes women's political rights, even briefly, is in scope and should be classified by that argument.

Use the CONTEXT (preceding and following speech turns) to resolve references to "the Bill" or "this question," to identify arguments the speaker is responding to, and to catch irony or implicit references. Classify only the TARGET; stances expressed only in the CONTEXT do not count toward the TARGET's stance. If the TARGET text contains contributions from multiple speakers (Hansard sometimes concatenates a parliamentary question with its reply, or an interjection mid-speech), classify the TARGET as a whole.

Labels. Classify the speaker's stance as:

- "for": supports women's political rights or representation.
- "against": opposes women's political rights or representation.
- "both": the speech makes substantive arguments on both sides of women's political rights. Most often this takes the form of supporting rights for one subset of women while opposing them for another (married vs single women; women with property vs working women; women over 30 vs women under 30), or supporting one aspect of political participation while opposing another (voting vs holding office; local councils vs Parliament; jury service vs franchise), or pairing substantive endorsement of the cause with substantive opposition to its current form on procedural or constitutional grounds.
- "irrelevant": the speech does not argue about women's political rights, even if it mentions them in passing.

Distinguishing "both" from "for" or "against" with caveats. Reserve "both" for genuine two-sidedness with substantive argument on each side. A speaker who supports women's political rights in principle but objects to this specific bill or mechanism is "for" (procedural objection is not substantive opposition). A speaker who opposes women's political rights but acknowledges the strength of the case for them is "against" (grudging acknowledgement is not endorsement). A speaker who supports with caveats about timing or implementation is "for" (caveats are not opposition).

USER
TARGET:
{target_text}

CONTEXT:
{context_text}
```
