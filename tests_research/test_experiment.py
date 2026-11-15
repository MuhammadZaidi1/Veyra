"""Tiny synthetic training checks; all temporary checkpoints are removed."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from veyra3d.checkpoint import load_checkpoint, save_checkpoint
from veyra3d.config import DataConfig, ExperimentConfig, ModelConfig
from veyra3d.diagnostics import smoke
from veyra3d.experiment import evaluate, train
from veyra3d.inference import Predictor
from veyra3d.runtime import initialize


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='veyra3d-engine-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = ExperimentConfig(model=ModelConfig(64,32),
                                       data=DataConfig(train_samples=4, val_samples=2, test_samples=2),
                                       epochs=1, batch_size=2, run_dir=str(self.root/'run'))

    def test_short_training_evaluation_and_memory_only_inference(self):
        result = train(self.config, quiet=True)
        self.assertEqual(result['global_step'], 2)
        state = load_checkpoint(self.root/'run/last.pt')
        self.assertTrue(state['synthetic_only'])
        self.assertNotIn('images', state)
        self.assertNotIn('vertices', state)
        self.assertTrue(all(value.device.type=='cpu' for value in state['model'].values()))
        validation = evaluate(self.config, self.root/'run/last.pt', 'val')
        held_out = evaluate(self.config, self.root/'run/last.pt', 'test')
        self.assertNotEqual(validation['dataset_signature'], held_out['dataset_signature'])
        self.assertNotEqual(validation['metrics']['relative_vertex_error'], held_out['metrics']['relative_vertex_error'])
        self.assertEqual(validation['metrics']['optimizer_steps'], 0)
        self.assertTrue(validation['synthetic_checkpoint'])
        # Evaluation of a training config that requests AMP still uses float32.
        evaluation_from_amp_config=evaluate(replace(self.config,amp=True),self.root/'run/last.pt')
        self.assertEqual(validation['metrics']['relative_vertex_error'],
                         evaluation_from_amp_config['metrics']['relative_vertex_error'])
        prediction = Predictor(self.root/'run/last.pt')
        vertices = prediction.predict(Image.new('RGB',(50,80),(40,80,120)))
        self.assertTrue(prediction.synthetic_only)
        self.assertEqual(vertices.shape,(468,3))
        self.assertTrue(np.isfinite(vertices).all())
        self.assertEqual({p.name for p in (self.root/'run').iterdir()},
                         {'.gitignore','experiment.json','last.pt','best.pt','metrics.jsonl'})

    def test_epoch_boundary_resume_matches_uninterrupted_cpu_weights_exactly(self):
        whole_config = replace(self.config, epochs=2, run_dir=str(self.root/'whole'))
        train(whole_config, quiet=True)
        train(self.config, quiet=True)
        resumed = train(replace(self.config, epochs=2), self.root/'run/last.pt', quiet=True)
        continuous = load_checkpoint(self.root/'whole/last.pt')
        restored = load_checkpoint(self.root/'run/last.pt')
        self.assertEqual(resumed['global_step'], continuous['global_step'])
        for name in continuous['model']:
            torch.testing.assert_close(continuous['model'][name], restored['model'][name], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_epoch_boundary_resume_matches_uninterrupted_cuda_weights_exactly(self):
        self.config = replace(self.config, device='cuda')
        self.test_epoch_boundary_resume_matches_uninterrupted_cpu_weights_exactly()
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_changed_training_or_dataset_signature_rejects_resume(self):
        train(self.config, quiet=True)
        with self.assertRaisesRegex(ValueError,'config changed'):
            train(replace(self.config, epochs=2, learning_rate=.1), self.root/'run/last.pt', quiet=True)
        state = load_checkpoint(self.root/'run/last.pt')
        state['data_signatures']['train']='changed content'
        save_checkpoint(self.root/'mutated.pt',state)
        with self.assertRaisesRegex(ValueError,'contents changed'):
            train(replace(self.config, epochs=2), self.root/'mutated.pt', quiet=True)

    def test_changed_source_signature_rejects_resume(self):
        train(self.config,quiet=True)
        state=load_checkpoint(self.root/'run/last.pt')
        state['source_signature']='changed code'
        save_checkpoint(self.root/'changed-source.pt',state)
        with self.assertRaisesRegex(ValueError,'source changed'):
            train(replace(self.config,epochs=2),self.root/'changed-source.pt',quiet=True)

    def test_partial_gradient_accumulation_group_is_not_dropped(self):
        config=replace(self.config, data=DataConfig(train_samples=5,val_samples=2,test_samples=2), accumulation_steps=2)
        result=train(config,quiet=True)
        self.assertEqual(result['history'][0]['train']['samples'],5)
        self.assertEqual(result['global_step'],2)  # Three microbatches: a pair, then one.

    def test_run_folder_overwrite_is_refused(self):
        run=self.root/'run';run.mkdir()
        marker=run/'keep.txt';marker.write_text('existing user file')
        with self.assertRaisesRegex(ValueError,'not empty'):
            train(self.config,quiet=True)
        self.assertEqual(marker.read_text(),'existing user file')

    def test_atomic_checkpoint_failure_keeps_previous_checkpoint(self):
        train(self.config,quiet=True)
        path=self.root/'run/last.pt'
        before=path.read_bytes()
        with mock.patch('veyra3d.checkpoint.torch.save',side_effect=RuntimeError('synthetic write failure')):
            with self.assertRaises(RuntimeError):
                save_checkpoint(path, {'model': {}})
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(list(path.parent.glob('.checkpoint-*')),[])

    def test_checkpoint_missing_or_unknown_schema_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing'):
            load_checkpoint(self.root/'absent.pt')
        torch.save({'not':'a Veyra model'},self.root/'bad.pt')
        with self.assertRaisesRegex(ValueError,'complete'):
            load_checkpoint(self.root/'bad.pt')

    def test_cpu_amp_is_rejected_instead_of_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'CUDA-only'):
            initialize(replace(self.config,amp=True))

    def test_local_manifest_pipeline_uses_only_generated_fixtures_and_detects_changed_data(self):
        from test_data import write_sample
        data_root=self.root/'data';data_root.mkdir()
        first=write_sample(data_root,'train','group-train',(20,30,40))
        second=write_sample(data_root,'val','group-val',(80,100,130))
        (data_root/'train.jsonl').write_text(json.dumps(first)+'\n')
        (data_root/'val.jsonl').write_text(json.dumps(second)+'\n')
        config=replace(self.config,data=DataConfig(kind='manifest',root=str(data_root)))
        result=train(config,quiet=True)
        self.assertEqual(result['global_step'],1)
        validation=evaluate(config,self.root/'run/last.pt')
        self.assertTrue(np.isfinite(validation['metrics']['relative_vertex_error']))
        with self.assertRaisesRegex(ValueError,'configured'):
            evaluate(config,self.root/'run/last.pt','test')
        Image.new('RGB',(40,80),(90,100,110)).save(data_root/'train.png')
        with self.assertRaisesRegex(ValueError,'contents changed'):
            train(replace(config,epochs=2),self.root/'run/last.pt',quiet=True)

    def test_spawnable_loader_worker_runs_synthetic_training(self):
        result=train(replace(self.config,workers=1),quiet=True)
        self.assertEqual(result['global_step'],2)
        self.assertEqual(result['history'][0]['train']['samples'],4)

    def test_smoke_completes_only_four_steps_and_marks_no_facial_accuracy(self):
        result=smoke('cpu')
        self.assertEqual(result['optimizer_steps'],4)
        self.assertEqual(result['epochs'],2)
        self.assertEqual(result['facial_accuracy'],'not established')
        self.assertTrue(np.isfinite(result['final_train_loss']))
        self.assertLess(result['final_train_loss'],result['initial_train_loss'])


class ExclusionTests(unittest.TestCase):
    def test_private_data_and_checkpoints_are_ignored_while_research_code_is_visible(self):
        root=Path(__file__).resolve().parents[1]
        if not (root/'.git').exists():
            self.skipTest('Git metadata absent')
        paths=['runs/a/experiment.json','somewhere/weights.pt','datasets/private/train.jsonl',
               'somewhere/private.jsonl','.venv-research/bin/python']
        result=subprocess.run(['git','check-ignore','--no-index',*paths],cwd=root,capture_output=True,text=True,check=False)
        self.assertEqual(set(result.stdout.splitlines()),set(paths))
        source=subprocess.run(['git','check-ignore','--no-index','veyra3d/model.py'],cwd=root,capture_output=True,check=False)
        self.assertEqual(source.returncode,1)
