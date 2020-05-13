from pyeparse.reading import Reading


class CustomReading(Reading):
    def _define_ffix_nogaze(self):
        max_pos = self._define_max_pos()
        data = self._data
        # initializing
        ffix_nogaze = np.zeros((len(self._data), 1), int)
        ffix_nogaze[0] = 1
        ffix_ias = np.zeros((self.n_ias, 1))
        ref_trial = data.iloc[0]['trial']

        for idx in range(1, len(data)):
            meas = data.iloc[idx]
            prev_meas = data.iloc[idx - 1]
            if meas['trial'] > ref_trial:
                ffix_ias = np.zeros((self.n_ias, 1), int)
                ref_trial = meas['trial']
            if ffix_ias[meas['fix_pos'].astype(int)] == 0:
                ffix_ias[meas['fix_pos'].astype(int)] = 1
                ffix_nogaze[idx] = 1

        return ffix_nogaze

    def get_ffix_nogaze_duration(self, ia):
        labels = self._data.columns.get_values()
        ffix_nogaze = self._define_ffix_nogaze()
        data = map(DataFrame, [ffix_nogaze])
        data.insert(0, self._data)
        labels = np.hstack((labels, ['ffix_nogaze']))
        data = concat(data, axis=1)
        data.columns = labels

        data = self._data[self._data['ffix_nogaze'] == 1]
        data = data[data['fix_pos'] == ia]
        data = data.groupby(by=['trial'], as_index=False).mean()
        data = data.reset_index(drop=True)

        return data
