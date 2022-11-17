.. _Guidelines:

Guidelines for gammasim-tools Developers
****************************************

This section provides help and guidelines for developers of gammasim-tools.
If you want to contribute to gammasim-tools, please use one of the contact points listed at the entry page of this documentation.
In general, please take note of the `ctapipe Development Guidelines <https://cta-observatory.github.io/ctapipe/development/index.html>`_.
gammasim-tools follows the same `style <https://cta-observatory.github.io/ctapipe/development/style-guide.html#>`_ and `code guidelines <https://cta-observatory.github.io/ctapipe/development/code-guidelines.html>`_ as `ctapip <https://github.com/cta-observatory/ctapipe/>`_.

Project setup
=============

The main code repository for gammasim-tools is on GitHub:

`https://github.com/gammasim/gammasim-tools <https://github.com/gammasim/gammasim-tools>`_

The main directories for developers are the `simtools <https://github.com/gammasim/gammasim-tools/tree/master/simtools>`_, `applications <https://github.com/gammasim/gammasim-tools/tree/master/applications>`_, `tests <https://github.com/gammasim/gammasim-tools/tree/master/tests>`_, and `docs <https://github.com/gammasim/gammasim-tools/tree/master/docs>`_ folders.


Python version
==============

The gammasim-tools package is currently developed for Python 3.9.


Code formatting
===============

Linting and code checks are done automatically using the pre-commit functionaility using ``isort``, ``black`` and ``pyflakes``. As part of the CI workflow Codacy performs a few additional code checks as well.

It is recommended for developers to install ``pre-commit``:

.. code-block::

    pre-commit install

The configuration of ``pre-commit`` is defined in `.pre-commit-config.yaml <https://github.com/gammasim/gammasim-tools/blob/master/.pre-commit-config.yaml>`_.

For testing, pre-commit can be applied locally without commit:

.. code-block::

    pre-commit run --all-files

In rare cases, one might want to skip pre-commit checks with

.. code-block::

    git commit --no-verify

Logging
=======

Sufficient logging information should be provided to users and developers. As general guideline, the following logging levels should be used:

- **INFO**: information useful for the general user about the progress, intermediate results, input or output.
- **WARNING**: information for the general user or developer on something they should know but cannot change.
- **DEBUG**: information only interesting for developers or useful for debugging.
- **ERROR**: something which always leads to an exception or program exit.

Use ``logger.error, logger.warning, logger.debug, logger.info``.


Testing
=======

pytest framework is used for unit testing.
The test modules are located in `simtools/tests <https://github.com/gammasim/gammasim-tools/tree/master/tests>`_ modules separated by unit and integration tests.
Every module should have a reasonable unit test, ideally all functions should be covered by tests.
Applications should be tested using integration tests.
It is important to write the tests in parallel with the modules
to assure that the code is testable.

General service functions for tests (e.g., DB connection) can be found in `conftest.py <https://github.com/gammasim/gammasim-tools/blob/master/tests/conftest.py>`_. This should be used to avoid duplication.


.. note:: Developers should expect that code changes affecting several modules are acceptable in case unit tests are successful.


Documentation
=============

Sphinx is used to create this documentation from the files in the `docs <https://github.com/gammasim/gammasim-tools/tree/master/docs>`_ directory and from the docstrings in the code.
This is done automatically with each merge into the master branch, see the `GitHub Action workflow CI-docs <https://github.com/gammasim/gammasim-tools/blob/master/.github/workflows/CI-docs.yml>`_.

Docstrings following the Numpy style must be added to any public function, class or method.
It is also recommended to add docstrings-like comments to private functions.

In the application, the modules should contain docstrings with a general description, command line
parameters, and examples.
A typical example should look like:

.. code-block:: python

    def a_function(parameter):
        """
        Description of what the function is doing.

        Parameters
        ----------
        parameter: type
            description of parameters

        Returns
        -------
        describe return values

        Raises
        ------
        describe exceptions raised

        """

        ...code


For a reference of the numpydoc format, see https://numpydoc.readthedocs.io/en/latest/format.html

For writing and testing documentation locally:

.. code-block::

    cd docs
    make html

This is especially recommended to identify warnings and errors by Sphinx (e.g., from badly formatted docstrings or RST files).
The documentation can be viewed locally in a browser starting from the file ``./docs/build/html/index.html``.


Writing Applications
====================

Applications are command lines tools that should be build off of the simtools library.
Application should not include complex algorithm, this should be done at the module level.

All applications should follow the same structure:


.. code-block:: python

    def main():

        # application name
        label = Path(__file__).stem
        # short description of the application
        description = "...."
        # short help on how to use the application
        usage = "....."

        # configuration handling (from command line, config file, etc)
        config = Configurator(label=label, description=description, usage=usage)
        ...
        args_dict, db_dict = config.initialize()

        # generic logger
        logger = logging.getLogger()
        logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

        # application code follows
        ...

Application handling should be done using the ``Configurator`` class, which allows to set configurations from command line, configuration file, or environmental variables.


Dependencies
============

Dependencies on python packages are listed in the `environment file <https://github.com/gammasim/gammasim-tools/blob/master/environment.yml>`_.
Some of the packages installed are used for the development only and not needed for executing gammasim-tools applications.


Integration with CORSIKA and sim_telarray
=========================================

CORSIKA and sim_telarray are external tools to simtools.
Their integration should be
minimally coupled with the rest of the package. The modules that depend directly on these
tools should be connected to the rest of the package through interfaces. This way, it
will be easier to replace these tools in the future.

One example of this approach is `simulator module <https://github.com/gammasim/gammasim-tools/blob/master/simtools/simulator.py>`_,
which connects to the tools used to manage and run simulations.


Handling data files
===================

.. warning:: Requires review

Data files should be kept outside of the gammasim-tools repository.
Some auxiliary files can be found in the `data directory <https://github.com/gammasim/gammasim-tools/tree/master/data>`_.
Note that this is under review and might go away in near future.


Naming
======

Telescope Names
---------------

The telescope names as used by gammasim-tools follow the pattern "Site-Class-Type", where:

* "Site" is either "North" or "South";
* "Class" is either "LST", "MST", "SCT" or "SST";
* "Type" is a single number ONLY in case of a real telescope existing at the site or a string containing a "D" in case of any other telescope design.

For example:

* "North-LST-1" is the first LST commissioned at the La Palma site, while "North-LST-D234" is the current design of the further 3 LSTs.
* "North-MST-FlashCam-D" and "North-MST-NectarCam-D" are the two MST designs containing different cameras.

Any input telescope names can (and should) be validated by the function validateTelescopeName (see module :ref:`util.names <utilnames>`).
For the Site field, any different capitalization (e.g "south") or site names like "paranal" and "lapalma" will be accepted
and converted to the standard ones. The same applies to the Class field.
For the Type field, any string will be accepted and a selected list of variations will be converted to the standard ones
(e.g "flashcam" will be converted to "FlashCam").


Validating names
----------------

Names that are recurrently used along the the package should be validated when given as input.
Examples of names are: telescope, site, camera, model version. The functionalities to validate names
are found in  :ref:`util.names <utilnames>`. The function validateName receives the input string and a name dictionary,
that is usually called allSomethingNames. This dictionary contain the possible names (as keys) and lists
of allowed alternatives names as values. In case the input name is found in one of the lists, the key
is returned.

The name dictionaries are also defined in util.names. One should also define specific functions named
validateSomethingNames that call the validateName with the proper name dictionary. This is only meant to
provide a clear interface.

This is an example of a name dictionary:


.. code-block::

  all_site_names = {
    "South": ["paranal", "south"],
    "North": ["lapalma", "north"]
  }

And this is an example of how the site name is validated in the :ref:`telescope_model` module:


.. code-block:: python

  self.site = names.validate_site_name(site)

where site was given as parameter to the ``TelescopeModel::__init__`` function.



Input validation
================

.. warning:: Requires review

Any module that receives configurable inputs (e.g. physical parameters)
must have them validated. The validation assures that the units, type and
format are correct and also allow for default values.

The configurable input must be passed to classes through a dictionary or a yaml
file. In the case of a dictionary the parameter is called configData, and in the
case of a yaml file, configFile. See the ray_tracing module for an example.

The function gen.collectDataFromYamlOrDict(configData, configFile, allowEmpty=False)
must be used to read these arguments. It identifies which case was given and
reads it accordingly, returning a dictionary. It also raises an exception in case none are
given and not allowEmpty.

The validation of the input is done by the function gen.validateConfigData, which
receives the dictionary with the collected input and a parameter dictionary. The parameter
dictionary is read from a parameter yaml file in the data/parameters directory.
The file is read through the function io.getDataFile("parameters", filename)
(see data files section).

The parameter yaml file contains the list of parameters to be validated and its
properties. See an example below:

.. code-block:: yaml

  zenithAngle:
    len: 1
    unit: !astropy.units.Unit {unit: deg}
    default: !astropy.units.Quantity
      value: 20
      unit: !astropy.units.Unit {unit: deg}
    names: ['zenith', 'theta']


* len gives the length of the input. If null, any len is accepted.
* unit is the astropy unit
* default must have the same len
* names is a list of acceptable input names. The key in the returned dict will have the name given at the definition of the block (zenithAngle in this example)


Docker Container for Development
=================================

A docker container is made available for developers, see the `gammasim-tools container repository <https://github.com/gammasim/containers/tree/main/dev>`_.
The container has the python packages, CORSIKA, and sim_telarray pre-installed.
Setting up a system to run gammasim-tools applications or tests should be a matter of minutes:

\1. install Docker and start the Docker application (see `Docker installation page <https://docs.docker.com/engine/install/>`_). Other container systems like Apptainer, Singularity, Buildah/Podman, etc should work, but are not thoroughly tested.

2. obtain the access parameters for the CTA Simulation Model data base and write a small script ``set_DB_environ.sh`` to set these parameters to be used in the container:
.. code-block::
    export DB_API_USER=<db_user_name>
    export DB_API_PW=<db_password>
    export DB_API_PORT=<db_port>
    export DB_SERVER=<db_server>

3. Start up a container and e.g. run the gammasim-tools unit tests using the following commands:
.. code-block::

    # create a working directory
    mkdir external && cd external
    # clone gammasim-tools repository
    git clone https://github.com/gammasim/gammasim-tools.git
    # startup a container (download if is not available in your environment)
    docker run --rm -it -v "$(pwd)/external:/workdir/external" ghcr.io/gammasim/containers/gammasim-tools-dev:v0.3.0-dev1 bash -c "$(cat ./entrypoint.sh) && bash"
    # Now you can run gammasim-tools application
    # Try e.g. to run the unit tests:
    pytest tests/unit_tests/
