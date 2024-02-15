import 'dart:convert';

import 'package:csv/csv.dart';
import 'package:normal/normal.dart';
import 'dart:collection';
import 'dart:math';
import 'dart:io';

abstract class Individual {
  double fitness(List<List<double>> X, List y);
  Individual mutate();
  dynamic predict(List<double> input);
  double getDifference(Individual other);
}

class AbaloneIndividual implements Individual {
  static const int _genesCount = 8;
  List<double> genes = [];

  Random rng = Random();

  AbaloneIndividual(this.genes) {
    if (genes.length != _genesCount) {
      throw UnsupportedError("Trying to assign for new individual chromosome that have different count of genes(${genes.length} must be $_genesCount)");
    }
  }

  AbaloneIndividual.random() {
    for (int i = 0; i < _genesCount; i ++) {
      genes.add(rng.nextDouble());
    }
  }

  double _absolute_error(double y, double y_predicted) {
    return (y - y_predicted).abs();
  }

  @override
  double predict(List<double> input) {
    double prediction = 0.0;

    for (int i = 0; i < input.length; i ++) {
      prediction += genes[i] * input[i];
    }

    return prediction + genes[_genesCount - 1];
  }

  @override
  double fitness(List<List<double>> X, List y) {
    if (X.length != y.length) {
      throw UnsupportedError("Length of input data must be the same as length of labels.");
    }

    double error = 0;
    for (int i = 0; i < X.length; i ++) {
      error += _absolute_error(y[i], predict(X[i]));
    }
    error /= X.length;

    return 1 / (error + 1);
  }

  @override
  Individual mutate() {
    Normal nrng = Normal();
    List<double> newGenes = [];

    for (int i = 0; i < _genesCount; i ++) {
      newGenes.add(genes[i] + nrng.generate(1)[0]);
    }

    return AbaloneIndividual(newGenes);
  }

  @override
  double getDifference(Individual other) {
    AbaloneIndividual otherAbalone = other as AbaloneIndividual;
    List<double> otherGenes = otherAbalone.genes;

    double distance = 0.0;
    for (int i = 0; i < _genesCount; i ++) {
      distance += (genes[i]/(otherGenes[i] + 0.01)).abs();
    }

    return distance;
  }
}

abstract class IndividualFactory {
  Individual createIndividualRandomly();
}

class AbaloneIndividualFactory implements IndividualFactory {
  @override
  Individual createIndividualRandomly() {
    return AbaloneIndividual.random();
  }
}

abstract class StopCriterion {
  bool toStop(List<Individual> newPopulation, List<Individual> oldPopulation, int generation);
}

class StopOnMaxGeneration implements StopCriterion {
  final int _maxgen;
  StopOnMaxGeneration(this._maxgen) {
    if (_maxgen < 0) {
      throw UnsupportedError("maximal generation must be at least zero(>= 0).");
    }
  }

  @override
  bool toStop(List newPopulation, List oldPopulation, int generation) {
    return generation >= _maxgen;
  }
}

class StopOnConvergenceByRatio implements StopCriterion {
  final double _difference;
  
  StopOnConvergenceByRatio(this._difference) {
    if (_difference < 0) {
      throw UnsupportedError("Difference must be at least zero(>= 0).");
    }
  }

  @override
  bool toStop(List<Individual> newPopulation, List<Individual> oldPopulation, int generation) {
    print(newPopulation[0].getDifference(newPopulation[newPopulation.length - 1]));

    return newPopulation[0].getDifference(newPopulation[newPopulation.length - 1])
     - oldPopulation[0].getDifference(oldPopulation[oldPopulation.length - 1]) < _difference;
  }
}

class EvolutionaryProgramming {
  List<Individual> population = [];

  int _generation = 0;

  int offspringCount;
  int parentsCount;
  // double mutationDeviation;

  IndividualFactory factory;
  StopCriterion criterion;

  EvolutionaryProgramming({
    required this.offspringCount,
    required this.parentsCount,
    required this.factory,
    required this.criterion,
  }) {
    if (this.offspringCount < 1) {
      throw UnsupportedError("Offspring count may not be less than one.");
    }

    if (this.parentsCount < 1) {
      throw UnsupportedError("Parents count may not be less than one.");
    }

    // if (this.mutationDeviation <= 0) {
    //   throw UnsupportedError("Mutation deviation can't be less than one.");
    // }

    for (int i = 0; i < parentsCount; i++) {
      population.add(factory.createIndividualRandomly());
    }
  }

  UnmodifiableListView<Individual> evolve(List<List<double>> X, List<double> y, {bool verbose = false}) {
    bool timeToStop = false;

    population.sort((left, right) => (right.fitness(X, y) - left.fitness(X, y)).sign.toInt());
    while (! timeToStop) {
      List<Individual> newPopulation = [];
      population.forEach((element) {
        newPopulation.add(element.mutate());
        // newPopulation.add(element + mutationDeviation * normalRng.generate(1)[0]);
      });

      newPopulation.addAll(population);
      newPopulation.sort(
          (left, right) => (right.fitness(X, y) - left.fitness(X, y)).sign.toInt());
      newPopulation = newPopulation.sublist(0, parentsCount);
      // for (int i = 0; i < newPopulation.length; i ++) {
      //   print("$i = ${evaluate(newPopulation[i])}, ");
      // }

      // ============
      _generation++;

      if (verbose) {
        print("$_generation max fitness: ${newPopulation[0].fitness(X, y)}");
      }

      timeToStop = criterion.toStop(newPopulation, population, _generation);
      population = newPopulation; 
    }

    return UnmodifiableListView(population);
  }
}

void main() async {
  const trainDataPath = "./abalone.data";

  final input = File(trainDataPath).openRead();
  final data = await input
      .transform(utf8.decoder)
      .transform(CsvToListConverter(fieldDelimiter: ','))
      .toList();

  List<List<double>> X = [];
  List<double> y = [];
  for (final row in data) {
    List inputData = row.sublist(0, 8); 
    // X.add(row.sublist(0, 8));
    String sex = inputData[0];
    switch(sex) {
      case 'M':
        inputData[0] = 0.0;
        break;

      case 'F':
        inputData[0] = 1.0;
        break;

      case 'I':
        inputData[0] = 2.0;
        break;
    }

    X.add(inputData.map((val) => double.parse(val.toString())).toList());
    y.add(double.parse(row[8].toString()));
  }
  // print(y);

  EvolutionaryProgramming ep = EvolutionaryProgramming(
      offspringCount: 10,
      parentsCount: 10,
      factory: AbaloneIndividualFactory(),
      criterion: StopOnConvergenceByRatio(0.001)
    );

  UnmodifiableListView solutions = ep.evolve(X, y, verbose: true);

  for (int i = 0; i < solutions.length; i++) {
    Individual solution = solutions[i];
    print("Solution #$i: $solution");

    for (int j = 0; j < X.length; j ++) {
      print("${X[j]} -> ${solution.predict(X[j])} | ${y[j]}");
    }
  }
}
