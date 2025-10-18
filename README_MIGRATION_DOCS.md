# Migration Documentation Guide

## Overview

This folder contains three comprehensive documentation files designed to enable AI agents (like Claude Code, Copilot, or Codex) or human engineers to understand and execute the migration of the Deep Video Stabilization project.

---

## The Three Documents

### 1. **PROJECT_CONTEXT.md** (Comprehensive Reference)
**Purpose**: Complete understanding of the existing codebase

**When to Use**: 
- When you need to understand how the code works
- Before making any changes
- When debugging issues
- When planning modifications

**What's Inside**:
- Project overview and architecture
- Detailed code structure (file-by-file breakdown)
- Data flow and neural network architecture
- Loss functions explained
- FlowNet2 integration details (what needs replacing)
- Data formats and configurations
- Performance characteristics
- Edge cases and known issues

**Size**: ~1,200 lines | ~15 minutes to read

**Key Sections**:
- Section 2: Architecture Deep Dive
- Section 5: FlowNet2 Integration (critical for migration)
- Section 6: Code Structure
- Section 12: Context for AI Agents

---

### 2. **MIGRATION_REQUIREMENTS.md** (Technical Specification)
**Purpose**: Exact technical requirements for the migration

**When to Use**:
- When starting the migration
- To understand what needs to change
- To verify success criteria
- To troubleshoot issues

**What's Inside**:
- Detailed goals and requirements
- Python 3.10 + PyTorch 2.2 specifications
- RAFT integration guide (with code templates)
- PyTorch API update requirements
- Testing and validation procedures
- Acceptance criteria
- Rollback plan
- Timeline estimates

**Size**: ~1,000 lines | ~12 minutes to read

**Key Sections**:
- R1: Environment and Dependencies
- R2: FlowNet2 Replacement (with complete RAFT integration code)
- R3: PyTorch API Updates
- R5: Testing and Validation
- Acceptance Criteria

---

### 3. **TASK_CHECKLIST.md** (Execution Roadmap)
**Purpose**: Step-by-step actionable checklist

**When to Use**:
- During execution (your daily task list)
- To track progress
- To verify each step is complete
- To ensure nothing is missed

**What's Inside**:
- 40 granular tasks organized in 7 phases
- Exact commands to run
- Verification steps for each task
- Success criteria
- Dependencies between tasks
- Troubleshooting guide

**Size**: ~800 lines | ~10 minutes to read

**Task Breakdown**:
- Phase 1: Environment Setup (5 tasks, 2 hours)
- Phase 2: RAFT Integration (7 tasks, 4 hours)
- Phase 3: Remove FlowNet2 (4 tasks, 1 hour)
- Phase 4: Update PyTorch APIs (8 tasks, 4 hours)
- Phase 5: Testing (6 tasks, 8 hours)
- Phase 6: Documentation (5 tasks, 4 hours)
- Phase 7: Polish - Optional (5 tasks, 4 hours)

**Total**: ~27 hours for complete migration

---

## How to Use These Documents

### For AI Agents (Claude, Copilot, Codex)

**Recommended Prompt**:

```
I need you to migrate the Deep Video Stabilization project. I've prepared three 
comprehensive documentation files for you:

1. PROJECT_CONTEXT.md - Explains the entire codebase architecture
2. MIGRATION_REQUIREMENTS.md - Specifies what needs to change
3. TASK_CHECKLIST.md - Provides step-by-step tasks

Please:
1. First, read PROJECT_CONTEXT.md to understand the project
2. Then, read MIGRATION_REQUIREMENTS.md to understand the requirements
3. Finally, execute tasks from TASK_CHECKLIST.md in order

Start by reading PROJECT_CONTEXT.md and confirming you understand:
- The current architecture
- How FlowNet2 is integrated (Section 2.5)
- What needs to be changed

Then proceed with Phase 1 of TASK_CHECKLIST.md.
```

**Alternative Prompt (If context window limited)**:

```
I need to migrate Deep Video Stabilization from Python 3.6 + PyTorch 1.0 + FlowNet2 
to Python 3.10 + PyTorch 2.2 + RAFT.

Key changes needed:
1. Replace FlowNet2 with RAFT (no CUDA compilation)
2. Update deprecated PyTorch APIs (Variable, upsample_bilinear)
3. Change flow format from .flo to .npy
4. Update requirements.txt

I have three detailed documentation files:
- PROJECT_CONTEXT.md (architecture reference)
- MIGRATION_REQUIREMENTS.md (technical specs)
- TASK_CHECKLIST.md (step-by-step tasks)

Please start with TASK_CHECKLIST.md Phase 1 (Environment Setup) and execute 
each task, verifying completion before moving to the next.

Read the relevant sections from PROJECT_CONTEXT.md and MIGRATION_REQUIREMENTS.md 
as needed for context.
```

### For Human Engineers

**Recommended Workflow**:

**Day 1: Understanding**
1. Read PROJECT_CONTEXT.md (Section 1-7)
   - Focus on Section 2.5 (FlowNet2 Integration)
   - Understand Section 3 (Code Structure)
2. Skim MIGRATION_REQUIREMENTS.md
   - Focus on Section R2 (RAFT Integration)
3. Review TASK_CHECKLIST.md
   - Understand the 7 phases
   - Note estimated times

**Day 2-3: Setup and RAFT Integration**
1. Execute TASK_CHECKLIST.md Phase 1 (Environment)
2. Execute TASK_CHECKLIST.md Phase 2 (RAFT Integration)
3. Refer to MIGRATION_REQUIREMENTS.md Section R2 for code templates

**Day 4: Cleanup**
1. Execute Phase 3 (Remove FlowNet2)
2. Execute Phase 4 (Update APIs)
3. Refer to MIGRATION_REQUIREMENTS.md Section R3 for specifics

**Day 5-6: Testing**
1. Execute Phase 5 (Testing)
2. Compare results with original
3. Use PROJECT_CONTEXT.md Section 10 for validation

**Day 7: Documentation**
1. Execute Phase 6 (Documentation)
2. Update README.md
3. Create CHANGELOG.md

**Day 8 (Optional): Polish**
1. Execute Phase 7 (Polish)
2. Add type hints
3. Improve error handling

---

## Quick Reference

### Key Files to Modify

From MIGRATION_REQUIREMENTS.md and TASK_CHECKLIST.md:

**Critical Changes**:
1. `dvs/requirements.txt` - Update dependencies
2. `dvs/dataset.py` - Remove FlowNet2, load .npy flows
3. `dvs/model.py` - Remove Variable wrapper
4. `dvs/loss.py` - Replace upsample_bilinear
5. `dvs/train.py` - Update yaml.load
6. `dvs/inference.py` - Update yaml.load
7. `dvs/gyro/gyro_function.py` - Remove Variable

**New Files to Create**:
1. `dvs/flow_estimator.py` - RAFT wrapper class
2. `dvs/generate_flow.py` - Flow generation script
3. `dvs/tests/test_migration.py` - Test suite
4. `CHANGELOG.md` - Document changes

**Files to Delete**:
1. `dvs/flownet2/` - Entire directory

### Key Commands

**Environment Setup**:
```bash
conda create -n dvs python=3.10
pip install -r requirements.txt
```

**RAFT Setup**:
```bash
git clone https://github.com/princeton-vl/RAFT.git
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/raft-things.pth
```

**Flow Generation**:
```bash
python generate_flow.py --data_dir ./video
```

**Test Inference**:
```bash
python inference.py --config ./conf/stabilzation.yaml --dir_path ./video
```

### Success Criteria

**Minimum Viable Product**:
- ✅ `pip install -r requirements.txt` works
- ✅ Inference completes without errors
- ✅ Output video created
- ✅ No FlowNet2 references

**Complete Migration**:
- ✅ All deprecated APIs updated
- ✅ Tests pass
- ✅ Documentation updated
- ✅ Old checkpoints work

---

## Troubleshooting

### "Where do I start?"
→ Read PROJECT_CONTEXT.md Sections 1-2, then start TASK_CHECKLIST.md Phase 1

### "What exactly needs to change?"
→ Read MIGRATION_REQUIREMENTS.md Section R2 (FlowNet2 Replacement)

### "How do I integrate RAFT?"
→ MIGRATION_REQUIREMENTS.md R2.2 has complete code template

### "What if something breaks?"
→ TASK_CHECKLIST.md has verification steps after each task
→ MIGRATION_REQUIREMENTS.md has rollback plan

### "How long will this take?"
→ TASK_CHECKLIST.md shows 27 hours total (5-7 days)
→ Minimum viable product: 18 hours (2-3 days)

---

## Document Statistics

| Document | Lines | Words | Read Time | Purpose |
|----------|-------|-------|-----------|---------|
| PROJECT_CONTEXT.md | 1,200+ | 25,000+ | 15 min | Understand codebase |
| MIGRATION_REQUIREMENTS.md | 1,000+ | 20,000+ | 12 min | Technical specs |
| TASK_CHECKLIST.md | 800+ | 15,000+ | 10 min | Execute migration |
| **Total** | **3,000+** | **60,000+** | **37 min** | **Complete guide** |

---

## Examples for AI Agents

### Example 1: Complete Migration Prompt

```
Task: Migrate Deep Video Stabilization to modern stack

I've prepared three comprehensive documentation files for you in this repository:
1. PROJECT_CONTEXT.md - Full architecture and code explanation
2. MIGRATION_REQUIREMENTS.md - Technical requirements and specifications  
3. TASK_CHECKLIST.md - 40 step-by-step tasks across 7 phases

Your goal is to execute all tasks in TASK_CHECKLIST.md, starting with Phase 1.

Before you begin:
1. Read PROJECT_CONTEXT.md (at least sections 1, 2, and 5)
2. Skim MIGRATION_REQUIREMENTS.md (focus on R2: FlowNet2 Replacement)
3. Understand TASK_CHECKLIST.md structure

Then execute tasks sequentially:
- Phase 1: Environment Setup
- Phase 2: RAFT Integration (use code templates from MIGRATION_REQUIREMENTS.md R2.2)
- Phase 3: Remove FlowNet2
- Phase 4: Update PyTorch APIs
- Phase 5: Testing
- Phase 6: Documentation
- Phase 7: Polish (optional)

For each task:
1. Confirm you understand what needs to be done
2. Execute the task
3. Run verification commands
4. Mark task as complete

Refer back to PROJECT_CONTEXT.md and MIGRATION_REQUIREMENTS.md as needed for 
detailed context and specifications.

Start with Task 1.1: Create Python 3.10 Environment
```

### Example 2: Focused Task Prompt

```
I need help with a specific part of the migration:

Task: Replace FlowNet2 with RAFT for optical flow estimation

Context: I have three documentation files:
- PROJECT_CONTEXT.md explains how FlowNet2 is currently used (Section 2.5)
- MIGRATION_REQUIREMENTS.md has RAFT integration specs (Section R2)
- TASK_CHECKLIST.md has the specific tasks (Phase 2)

Please:
1. Read PROJECT_CONTEXT.md Section 2.5 to understand current FlowNet2 usage
2. Read MIGRATION_REQUIREMENTS.md Section R2 for RAFT integration details
3. Execute TASK_CHECKLIST.md Phase 2 (Tasks 2.1 - 2.7)

The key deliverables are:
- flow_estimator.py (RAFT wrapper class)
- generate_flow.py (flow generation script)
- Updated dataset.py (load .npy instead of .flo)

Use the code templates provided in MIGRATION_REQUIREMENTS.md R2.2 and R2.3.
```

### Example 3: Testing Focus Prompt

```
I've completed the migration implementation. Now I need comprehensive testing.

I have documentation:
- PROJECT_CONTEXT.md Section 10 (Testing and Validation)
- MIGRATION_REQUIREMENTS.md Section R5 (Testing)
- TASK_CHECKLIST.md Phase 5 (6 testing tasks)

Please execute all testing tasks:
1. Create test suite (Task 5.2)
2. Test checkpoint loading (Task 5.3)
3. Run full inference test (Task 5.4)
4. Compare output quality (Task 5.5)
5. Performance benchmark (Task 5.6)

For each test, run verification commands and report results.
Compare against success criteria in MIGRATION_REQUIREMENTS.md.
```

---

## Next Steps

1. **Choose your approach**:
   - AI Agent: Use one of the example prompts above
   - Human: Follow the 8-day workflow

2. **Start reading**:
   - Begin with PROJECT_CONTEXT.md (sections 1-2)
   - Skim MIGRATION_REQUIREMENTS.md
   - Review TASK_CHECKLIST.md structure

3. **Begin execution**:
   - Start TASK_CHECKLIST.md Phase 1
   - Mark tasks as complete: `- [ ]` → `- [x]`
   - Track progress in your copy of TASK_CHECKLIST.md

4. **Iterate**:
   - Verify each task before moving on
   - Refer back to documentation as needed
   - Update CHANGELOG.md with any changes

---

## Support

**Questions about the code?**
→ Check PROJECT_CONTEXT.md

**Questions about requirements?**
→ Check MIGRATION_REQUIREMENTS.md

**Questions about specific tasks?**
→ Check TASK_CHECKLIST.md

**Still stuck?**
→ Refer to the original paper or contact the authors

---

## Document Versions

- **Version**: 1.0
- **Created**: 2024
- **Purpose**: Migration from Python 3.6 + PyTorch 1.0 + FlowNet2 to Python 3.10 + PyTorch 2.2 + RAFT
- **Status**: Ready for execution

---

## License

These documentation files are provided as-is to assist with the migration of the 
Deep Video Stabilization project. The original code is subject to its own license 
(see project LICENSE file).

---

**Ready to start? Begin with PROJECT_CONTEXT.md Section 1-2, then proceed to TASK_CHECKLIST.md Phase 1!**

